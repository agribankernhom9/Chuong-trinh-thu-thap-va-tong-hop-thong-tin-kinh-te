# -*- coding: utf-8 -*-
# ============================================================
# CHƯƠNG TRÌNH THU THẬP VÀ TỔNG HỢP THÔNG TIN KINH TẾ VĨ MÔ
# CỦA VIỆT NAM TRÊN THỊ TRƯỜNG TÀI CHÍNH
# ============================================================

import os
import io
import math
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# (Luôn bật) OpenAI cho "AI phân tích và tư vấn"
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# ---------------------------
# Tham số & Từ điển hiển thị
# ---------------------------
WB_API_BASE = "https://api.worldbank.org/v2"
UNDP_API_BASE = "https://hdr.undp.org/sites/default/files/api/"
HDI_PSEUDO_CODE = "UNDP.HDI"

# Danh mục mặc định (code, EN name, unit)
DEFAULT_INDICATORS = [
    ("NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)", "%"),
    ("FP.CPI.TOTL.ZG", "Inflation, CPI (annual %)", "%"),
    ("SL.UEM.TOTL.ZS", "Unemployment (% labor force)", "%"),
    ("NE.EXP.GNFS.ZS", "Exports of goods & services (% of GDP)", "%"),
    ("NE.IMP.GNFS.ZS", "Imports of goods & services (% of GDP)", "%"),
    ("GC.DOD.TOTL.GD.ZS", "Central government debt (% of GDP)", "%"),
    ("BX.KLT.DINV.WD.GD.ZS", "FDI, net inflows (% of GDP)", "%"),
    ("SP.POP.TOTL", "Population (total)", "persons"),
    ("NY.GDP.PCAP.CD", "GDP per capita (current US$)", "USD"),
]

# Chỉ số mở rộng (SBV/IMF/GSO) — dùng proxy WB khi API gốc chưa sẵn (ghi rõ nguồn)
# Có thể thay thế bằng API chính thức khi bạn cung cấp endpoint/khóa.
EXTENDED_INDICATORS = [
    ("FR.INR.LEND", "Lãi suất cho vay (%)", "%", "WB proxy (IMF/GSO)"),
    ("FR.INR.DPST", "Lãi suất tiền gửi (%)", "%", "WB proxy (IMF/GSO)"),
    ("PA.NUS.FCRF", "Tỷ giá chính thức (LCU/USD)", "LCU/USD", "WB proxy (SBV)"),
    # Placeholder nếu muốn riêng lãi suất điều hành SBV: sẽ để trống khi chưa có API chính thức
    ("SBV.POLICY.RATE", "Lãi suất điều hành (SBV) (%)", "%", "SBV (placeholder)"),
]

VN_NAME_MAP = {
    "NY.GDP.MKTP.KD.ZG": ("Tăng trưởng GDP (năm)", "%"),
    "FP.CPI.TOTL.ZG": ("Lạm phát CPI (năm)", "%"),
    "SL.UEM.TOTL.ZS": ("Tỷ lệ thất nghiệp", "%"),
    "NE.EXP.GNFS.ZS": ("Xuất khẩu hàng hóa & dịch vụ", "% GDP"),
    "NE.IMP.GNFS.ZS": ("Nhập khẩu hàng hóa & dịch vụ", "% GDP"),
    "GC.DOD.TOTL.GD.ZS": ("Nợ chính phủ", "% GDP"),
    "BX.KLT.DINV.WD.GD.ZS": ("FDI, dòng vốn ròng", "% GDP"),
    "SP.POP.TOTL": ("Dân số", "người"),
    "NY.GDP.PCAP.CD": ("GDP bình quân đầu người", "USD"),
    HDI_PSEUDO_CODE: ("Chỉ số phát triển con người (HDI)", ""),
    # Extended:
    "FR.INR.LEND": ("Lãi suất cho vay", "%"),
    "FR.INR.DPST": ("Lãi suất tiền gửi", "%"),
    "PA.NUS.FCRF": ("Tỷ giá chính thức (LCU/USD)", "LCU/USD"),
    "SBV.POLICY.RATE": ("Lãi suất điều hành (SBV)", "%"),
}

AGRIBANK_RGB = "rgb(174,28,63)"   # #AE1C3F

def get_vn_label_with_unit(code: str) -> str:
    vn_name, vn_unit = VN_NAME_MAP.get(code, (None, None))
    if vn_name is None:
        en_unit, en_name = "", code
        for c, name, unit in DEFAULT_INDICATORS + [(x[0], x[1], x[2]) for x in EXTENDED_INDICATORS]:
            if c == code:
                en_name = name
                en_unit = unit
                break
        return f"{en_name} ({en_unit})" if en_unit else en_name
    return f"{vn_name} ({vn_unit})" if vn_unit else vn_name

def is_percent_unit(code: str) -> bool:
    _, unit = VN_NAME_MAP.get(code, (None, None))
    if unit is None:
        for c, _, u in DEFAULT_INDICATORS + [(x[0], x[1], x[2]) for x in EXTENDED_INDICATORS]:
            if c == code:
                unit = u
                break
    unit = (unit or "").lower()
    return "%" in unit

# ---------------------------
# Tối ưu: cache & song song
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def list_wb_countries():
    url = f"{WB_API_BASE}/country?format=json&per_page=400"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for item in data[1]:
        if item.get("region", {}).get("id") != "Aggregates":
            rows.append({"id": item["id"], "name": item["name"]})
    df = pd.DataFrame(rows).sort_values("name")
    return df

def _fetch_wb_indicator(country_code: str, indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    url = f"{WB_API_BASE}/country/{country_code}/indicator/{indicator_code}?date={start_year}:{end_year}&format=json&per_page=12000"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    if not isinstance(js, list) or len(js) < 2 or js[1] is None:
        return pd.DataFrame(columns=["Year", indicator_code])
    rows = []
    for rec in js[1]:
        y = rec.get("date")
        val = rec.get("value")
        try:
            y = int(y)
        except:
            continue
        rows.append({"Year": y, indicator_code: val})
    return pd.DataFrame(rows).sort_values("Year")

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_wb_indicators_parallel(country_code: str, indicator_codes: list, start_year: int, end_year: int) -> dict:
    out = {}
    with ThreadPoolExecutor(max_workers=min(8, len(indicator_codes) or 1)) as ex:
        futures = {ex.submit(_fetch_wb_indicator, country_code, code, start_year, end_year): code for code in indicator_codes}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                out[code] = fut.result()
            except Exception:
                out[code] = pd.DataFrame(columns=["Year", code])
    return out

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_undp_hdi(country_iso3: str, start_year: int, end_year: int) -> pd.DataFrame:
    try:
        url = f"{UNDP_API_BASE}v1/indicators/137506?countries={country_iso3}&years={start_year}-{end_year}"
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        js = r.json()
        data = js.get("data", [])
        rows = [{"Year": int(it.get("year")), HDI_PSEUDO_CODE: it.get("value")} for it in data if it.get("year") is not None]
        return pd.DataFrame(rows).sort_values("Year")
    except Exception:
        return pd.DataFrame(columns=["Year", HDI_PSEUDO_CODE])

def fetch_extended_indicator(country_code: str, code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Tải chỉ số mở rộng:
      - Nếu là SBV.POLICY.RATE: placeholder (trả DF rỗng) khi chưa tích hợp API SBV.
      - Còn lại: dùng World Bank proxy.
    """
    if code == "SBV.POLICY.RATE":
        return pd.DataFrame(columns=["Year", code])
    return _fetch_wb_indicator(country_code, code, start_year, end_year)

def merge_wide(dfs: list) -> pd.DataFrame:
    out = None
    for d in dfs:
        if d is None or d.empty:
            continue
        out = d if out is None else pd.merge(out, d, on="Year", how="outer")
    return (out.sort_values("Year").reset_index(drop=True)) if out is not None else pd.DataFrame(columns=["Year"])

def impute_missing(df: pd.DataFrame, method: str):
    if df.empty:
        return df, {}
    df2 = df.copy()
    report = {}
    numeric_cols = [c for c in df2.columns if c != "Year"]
    if method == "Giữ nguyên (N/A)":
        pass
    elif method == "Forward/Backward fill":
        df2[numeric_cols] = df2[numeric_cols].ffill().bfill()
    elif method == "Điền trung bình theo cột":
        for c in numeric_cols:
            df2[c] = df2[c].fillna(df2[c].mean(skipna=True))
    elif method == "Điền median theo cột":
        for c in numeric_cols:
            df2[c] = df2[c].fillna(df2[c].median(skipna=True))
    for c in numeric_cols:
        report[c] = int(df2[c].isna().sum())
    return df2, report


def _format_number_vn(val, decimals_auto=True, force_decimals=None):
    """Định dạng số kiểu Việt Nam: . tách nghìn, , tách thập phân.
    - force_decimals: nếu khác None thì luôn dùng số chữ số sau dấu phẩy này
    - nếu decimals_auto: |v|>=1000 -> 0; 1<=|v|<1000 -> 2; |v|<1 -> 3
    """
    import pandas as _pd
    if _pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return str(val)
    if force_decimals is not None:
        d = force_decimals
    elif decimals_auto:
        av = abs(v)
        if av >= 1000:
            d = 0
        elif av >= 1:
            d = 2
        else:
            d = 3
    else:
        d = 2
    s = f"{v:,.{d}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    stats = []
    numeric_cols = [c for c in df.columns if c != "Year"]
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            stats.append({"Chỉ tiêu": get_vn_label_with_unit(col), "Giá trị TB (Mean)": np.nan,
                          "Độ lệch chuẩn (Std)": np.nan, "Nhỏ nhất (Min)": np.nan, "Năm Min": None,
                          "Lớn nhất (Max)": np.nan, "Năm Max": None, "Trung vị (Median)": np.nan,
                          "Q1": np.nan, "Q3": np.nan, "Hệ số biến thiên (CV%)": np.nan})
            continue
        mean, std = s.mean(), s.std(ddof=1)
        min_val, max_val = s.min(), s.max()
        min_year = df.loc[df[col].idxmin(), "Year"] if not s.empty else None
        max_year = df.loc[df[col].idxmax(), "Year"] if not s.empty else None
        median, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
        cv = (std/mean*100.0) if mean and not math.isclose(mean, 0.0, abs_tol=1e-12) else np.nan
        stats.append({"Chỉ tiêu": get_vn_label_with_unit(col), "Giá trị TB (Mean)": mean, "Độ lệch chuẩn (Std)": std,
                      "Nhỏ nhất (Min)": min_val, "Năm Min": int(min_year) if pd.notna(min_year) else None,
                      "Lớn nhất (Max)": max_val, "Năm Max": int(max_year) if pd.notna(max_year) else None,
                      "Trung vị (Median)": median, "Q1": q1, "Q3": q3, "Hệ số biến thiên (CV%)": cv})
    return pd.DataFrame(stats)

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "Year"]
    return df[cols].corr(method="pearson") if len(cols) >= 2 else pd.DataFrame()

def add_trendline(df: pd.DataFrame, x: str, y: str):
    sub = df[[x, y]].dropna()
    if len(sub) < 2:
        return None
    a, b = np.polyfit(sub[x], sub[y], deg=1)
    x_line = np.linspace(sub[x].min(), sub[x].max(), 100)
    y_line = a * x_line + b
    return x_line, y_line, a, b

def to_excel_bytes(df_data: pd.DataFrame, df_stats: pd.DataFrame, corr: pd.DataFrame) -> bytes:
    import openpyxl  # ensure engine present
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_data.to_excel(writer, index=False, sheet_name="Data")
        df_stats.to_excel(writer, index=False, sheet_name="Stats")
        (corr if not corr.empty else pd.DataFrame()).to_excel(writer, sheet_name="Correlation")
    buf.seek(0)
    return buf.read()

# ---------------------------
# Giao diện (brand Agribank)
# ---------------------------
st.set_page_config(page_title="Chương trình thu thập & tổng hợp vĩ mô VN", layout="wide")

# CSS chủ đề Agribank + phong cách chuyển đổi số
st.markdown(f"""
<style>
:root {{ --brand: {AGRIBANK_RGB}; }}
.topbar {{
  width: 100%; padding: 8px 16px;
  background: linear-gradient(90deg, rgba(174,28,63,1) 0%, rgba(174,28,63,0.8) 60%, rgba(174,28,63,0.6) 100%);
  color: #fff; display: flex; align-items: center; gap: 12px; border-radius: 12px;
}}
.logo-chip {{ display: inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; background: rgba(255,255,255,0.15); font-weight:600; letter-spacing:0.4px; }}
.logo-dot {{ width:10px; height:10px; border-radius:50%; background:#fff; display:inline-block; }}
.main-title {{ text-align:center; color: #d80000; margin: 12px 0 2px 0; }}
.source-chips {{ display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-bottom:6px; }}
.source-chips .chip {{ padding:4px 10px; border-radius:999px; border:1px solid rgba(174,28,63,0.3); background: rgba(174,28,63,0.06); color:#7a0f28; font-size:12px; font-weight:600; }}
.stButton>button {{ background: var(--brand); color:#fff; border:0; border-radius:10px; padding:8px 16px; font-weight:700; }}
.stButton>button:hover {{ filter: brightness(1.05); }}
.stTabs [data-baseweb="tab"] {{ font-weight:700; color:#7a0f28; }}
[data-testid="stDataFrame"] {{ border: 1px solid rgba(174,28,63,0.25); border-radius:12px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, rgba(174,28,63,0.07), transparent); }}
</style>
""", unsafe_allow_html=True)

# Top bar
st.markdown("""
<div class="topbar">
  <div class="logo-chip"><span class="logo-dot"></span> AGRIBANK</div>
  <div style="font-weight:700; letter-spacing:0.3px;">Chuyển đổi số • Dữ liệu mở • Phân tích thông minh</div>
</div>
""", unsafe_allow_html=True)

# Tiêu đề chính (in HOA, căn giữa, đỏ)
st.markdown(
    "<h1 class='main-title'>CHƯƠNG TRÌNH THU THẬP VÀ TỔNG HỢP THÔNG TIN KINH TẾ VĨ MÔ CỦA VIỆT NAM TRÊN THỊ TRƯỜNG TÀI CHÍNH</h1>",
    unsafe_allow_html=True
)

# Chips nguồn dữ liệu
st.markdown("""
<div class="source-chips">
  <div class="chip">World Bank</div>
  <div class="chip">UNDP</div>
  <div class="chip">IMF</div>
  <div class="chip">GSO</div>
  <div class="chip">SBV</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar thiết lập
# ---------------------------
with st.sidebar:
    st.header("Thiết lập")

    # Quốc gia: mặc định hiển thị "VNM - Việt Nam"
    try:
        countries_df = list_wb_countries()
        labels, id_to_label = [], {}
        for _id, _name in zip(countries_df["id"], countries_df["name"]):
            label = "VNM - Việt Nam" if _id == "VNM" else f"{_id} — {_name}"
            labels.append(label)
            id_to_label[label] = (_id, "Việt Nam" if _id == "VNM" else _name)
        default_idx = labels.index("VNM - Việt Nam") if "VNM - Việt Nam" in labels else (int((countries_df["id"] == "VNM").idxmax()) if "VNM" in set(countries_df["id"]) else 0)
        country_label = st.selectbox("Quốc gia", options=labels, index=default_idx)
        sel_country, sel_country_name = id_to_label[country_label][0], id_to_label[country_label][1]
    except Exception:
        st.warning("Không lấy được danh sách quốc gia từ World Bank. Dùng mặc định: VNM - Việt Nam.")
        sel_country, sel_country_name = "VNM", "Việt Nam"

    st.subheader("Khoảng năm")
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        start_year = st.number_input("Từ năm", min_value=1960, max_value=2100, value=2000, step=1)
    with col_y2:
        end_year = st.number_input("Đến năm", min_value=1960, max_value=2100, value=2024, step=1)
    if start_year > end_year:
        st.error("Khoảng năm không hợp lệ: 'Từ năm' phải ≤ 'Đến năm'.")

    st.subheader("Chỉ số (World Bank)")
    indicator_map = {f"{name} [{code}]": code for code, name, _ in DEFAULT_INDICATORS}
    selection = st.multiselect(
        "Chọn chỉ số",
        options=list(indicator_map.keys()),
        default=[f"{name} [{code}]" for code, name, _ in DEFAULT_INDICATORS]
    )
    selected_codes = [indicator_map[o] for o in selection]

    # Chỉ số mở rộng (SBV/IMF/GSO)
    with st.expander("Chỉ số mở rộng (SBV / IMF / GSO)"):
        ext_map = {f"{label} [{code}] — nguồn: {src}": code for code, label, unit, src in EXTENDED_INDICATORS}
        ext_sel = st.multiselect(
            "Chọn chỉ số mở rộng",
            options=list(ext_map.keys()),
            # mặc định chọn các chỉ số WB proxy, trừ placeholder SBV
            default=[f"{label} [{code}] — nguồn: {src}" for code, label, unit, src in EXTENDED_INDICATORS if code != "SBV.POLICY.RATE"]
        )
        selected_ext = [ext_map[o] for o in ext_sel]

    if "missing_method" not in st.session_state:
        st.session_state["missing_method"] = "Giữ nguyên (N/A)"

# ---------------------------
# ETL: tải dữ liệu (song song)
# ---------------------------
with st.spinner("Đang lấy dữ liệu..."):
    wb_d = fetch_wb_indicators_parallel(sel_country, selected_codes, int(start_year), int(end_year))
    ext_d = {code: fetch_extended_indicator(sel_country, code, int(start_year), int(end_year)) for code in selected_ext}

dfs = list(wb_d.values()) + list(ext_d.values())

use_hdi = False
if use_hdi:
    with st.spinner("Đang lấy HDI từ UNDP..."):
        hdi_df = fetch_undp_hdi(sel_country, int(start_year), int(end_year))
        if not hdi_df.empty:
            dfs.append(hdi_df)

raw_df = merge_wide(dfs).copy()
has_missing = raw_df.drop(columns=["Year"]).isna().any().any() if not raw_df.empty else False

missing_method = st.session_state.get("missing_method", "Giữ nguyên (N/A)")
imputed_df, na_report = impute_missing(raw_df, missing_method)

stats_df = compute_descriptive_stats(imputed_df)
corr_df = correlation_matrix(imputed_df)

# ---------------------------
# Build DataFrame hiển thị & định dạng
# ---------------------------
def build_display_df(numeric_df: pd.DataFrame) -> pd.DataFrame:
    if numeric_df.empty:
        return numeric_df
    df = numeric_df.copy()
    rename_map = {c: get_vn_label_with_unit(c) for c in df.columns if c != "Year"}
    df_renamed = df.rename(columns=rename_map)

    # Định dạng số kiểu Việt Nam cho mọi cột (thêm % khi là đơn vị phần trăm)
    formatted = df_renamed.copy()
    inv = {get_vn_label_with_unit(code): code for code in df.columns if code != "Year"}
    for c in formatted.columns:
        if c == "Year":
            continue
        code0 = inv.get(c, None)
        is_pct = (code0 is not None) and is_percent_unit(code0) or ("%" in str(c))
        if is_pct:
            formatted[c] = formatted[c].apply(lambda v: (_format_number_vn(v, decimals_auto=False, force_decimals=2) + " %") if pd.notna(v) else "")
        else:
            formatted[c] = formatted[c].apply(lambda v: _format_number_vn(v) if pd.notna(v) else "")
    return formatted

display_df = build_display_df(imputed_df)

# ---------------------------
# Tabs
# ---------------------------
tab_data, tab_charts, tab_stats, tab_download, tab_ai = st.tabs([
    "📥 Dữ liệu",
    "📊 Biểu đồ",
    "📐 Thống kê mô tả",
    "⬇️ Tải dữ liệu",
    "🤖 AI phân tích và tư vấn"
])

with tab_data:
    st.subheader("Bảng số liệu đã thu thập")

    # Xử lý thiếu dữ liệu: chỉ hiện khi có thiếu
    if has_missing:
        with st.expander("⚠️ Thiếu dữ liệu phát hiện — Chọn phương án xử lý", expanded=False):
            st.selectbox(
                "Phương án xử lý",
                ["Giữ nguyên (N/A)", "Forward/Backward fill", "Điền trung bình theo cột", "Điền median theo cột"],
                index=["Giữ nguyên (N/A)", "Forward/Backward fill", "Điền trung bình theo cột", "Điền median theo cột"].index(
                    st.session_state.get("missing_method", "Giữ nguyên (N/A)")
                ),
                key="missing_method",
                help="Đổi lựa chọn sẽ tự áp dụng trong lần tải lại."
            )
            # (1) Thông báo N/A: hiển thị TÊN TIẾNG VIỆT (đúng như trong bảng)
            if any(v > 0 for v in na_report.values()):
                vn_badges = []
                for code, cnt in na_report.items():
                    if cnt > 0:
                        vn_badges.append(f"{get_vn_label_with_unit(code)}:{cnt}")
                if vn_badges:
                    st.info("Sau xử lý hiện tại vẫn còn N/A ở: " + ", ".join(vn_badges))

    # Cột STT bắt đầu từ 1
    if not display_df.empty:
        df_show = display_df.copy()
        if "Year" in df_show.columns:
            df_show = df_show.rename(columns={"Year": "Năm dữ liệu"})
        df_show.index = np.arange(1, len(df_show) + 1)
        df_show.index.name = "STT"
        st.dataframe(df_show, use_container_width=True, height=420)
    else:
        st.info("Chưa có dữ liệu để hiển thị.")

    # Trích nguồn
    source_list = ["World Bank Open Data"]
    if any(c in selected_ext for c in ["FR.INR.LEND", "FR.INR.DPST", "PA.NUS.FCRF"]):
        source_list.append("WB proxy cho chỉ số SBV/IMF/GSO")
    if "SBV.POLICY.RATE" in selected_ext:
        source_list.append("SBV (lãi suất điều hành — placeholder)")
    st.caption("Nguồn dữ liệu: " + "; ".join(source_list))

with tab_charts:
    st.subheader("Trực quan hoá")
    chart_types = st.multiselect("Chọn loại biểu đồ", ["Line", "Bar", "Combo", "Scatter", "Heatmap"], default=["Line", "Heatmap"])
    available_series = [c for c in imputed_df.columns if c != "Year"]
    selected_series_for_plot = st.multiselect(
        "Chọn chỉ tiêu cần vẽ",
        options=available_series,
        default=available_series,
        format_func=lambda code: get_vn_label_with_unit(code)
    )

    if imputed_df.empty or not selected_series_for_plot:
        st.info("Chưa đủ dữ liệu hoặc chưa chọn chỉ tiêu.")
    else:
        df_plot = imputed_df[["Year"] + selected_series_for_plot].copy()

        if "Line" in chart_types:
            st.markdown("**Biểu đồ đường — Xu hướng theo thời gian**")
            m = df_plot.melt(id_vars="Year", var_name="Indicator", value_name="Value")
            m["Indicator"] = m["Indicator"].apply(get_vn_label_with_unit)
            def _fmt_row(r):
                is_pct = "%" in str(r["Indicator"])
                val = _format_number_vn(r["Value"], decimals_auto=False, force_decimals=2) + " %" if is_pct else _format_number_vn(r["Value"])
                return f"{r['Indicator']}={val}<br>Năm={int(r['Year'])}"
            m["Hover"] = m.apply(_fmt_row, axis=1)
            import plotly.graph_objects as go
            import numpy as _np
            # Xây thủ công từng trace để đảm bảo hover đúng tên/giá trị và mặc định 1 line
            fig = go.Figure()
            _norm = lambda s: str(s).strip().lower()
            _default_name = get_vn_label_with_unit("NY.GDP.PCAP.CD")
            for ind_name in m["Indicator"].drop_duplicates().tolist():
                sub = m[m["Indicator"] == ind_name].copy()
                label_arr = _np.array([ind_name] * len(sub), dtype=object)
                year_arr = sub["Year"].astype(int).to_numpy()
                _is_pct = ("%") in str(ind_name)
                if "__val_fmt__" in sub.columns:
                    val_arr = sub["__val_fmt__"].to_numpy()
                else:
                    val_arr = sub["Value"].apply(lambda v: (_format_number_vn(v, decimals_auto=False, force_decimals=2) + " %") if _is_pct else _format_number_vn(v)).to_numpy()
                cd = _np.stack([label_arr, year_arr, val_arr], axis=-1)
                vis = True if _norm(ind_name) == _norm(_default_name) else "legendonly"
                fig.add_trace(go.Scatter(x=sub["Year"], y=sub["Value"], mode="lines+markers", name=ind_name,
                                         customdata=cd,
                                         hovertemplate="%{customdata[0]}=%{customdata[2]}<br>Năm=%{customdata[1]}<extra></extra>",
                                         visible=vis))
            fig.update_layout(height=450, legend_title_text="Chỉ tiêu", separators=",.", xaxis_title="Năm thu thập", yaxis_title="Giá trị")
            all_pct = m["Indicator"].astype(str).str.contains("%").all()
            fig.update_yaxes(tickformat=",.2f" if all_pct else ",.0f")
            fig.update_xaxes(tickformat="d")
            st.plotly_chart(fig, use_container_width=True)

        if "Bar" in chart_types:
            st.markdown("**Biểu đồ cột — So sánh theo năm**")
            bar_col = st.selectbox("Chỉ tiêu cho Bar", options=selected_series_for_plot, format_func=lambda c: get_vn_label_with_unit(c))
            df_bar = df_plot[["Year", bar_col]].copy()
            _is_pct = "%" in get_vn_label_with_unit(bar_col)
            df_bar["Hover"] = df_bar.apply(lambda r: f"{get_vn_label_with_unit(bar_col)}=" + (_format_number_vn(r[bar_col], decimals_auto=False, force_decimals=2) + " %" if _is_pct else _format_number_vn(r[bar_col])) + f"<br>Năm={int(r['Year'])}", axis=1)
            fig = px.bar(df_bar, x="Year", y=bar_col, title=get_vn_label_with_unit(bar_col), hover_data=None)
            fig.update_traces(hovertext=df_bar["Hover"], hovertemplate="%{hovertext}<extra></extra>")
            fig.update_layout(height=420, separators=",.", xaxis_title="Năm thu thập", yaxis_title="Giá trị")
            fig.update_yaxes(tickformat=",.2f" if _is_pct else ",.0f")
            fig.update_xaxes(tickformat="d")
            fig.update_layout(margin=dict(t=60,r=20,b=40,l=60), legend=dict(orientation="h", yanchor="top", y=-0.2), uniformtext_minsize=10, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)

        if "Combo" in chart_types:
            st.markdown("**Biểu đồ kết hợp — Bar + Line**")
            c1, c2 = st.columns(2)
            with c1:
                bar_c = st.selectbox("Bar =", options=selected_series_for_plot, format_func=lambda c: get_vn_label_with_unit(c), key="bar_combo")
            with c2:
                cand_line = [c for c in selected_series_for_plot if c != bar_c]
                line_c = st.selectbox("Line =", options=cand_line if cand_line else selected_series_for_plot,
                                      format_func=lambda c: get_vn_label_with_unit(c), key="line_combo")
            fig = go.Figure()
            _is_pct_bar = "%" in get_vn_label_with_unit(bar_c)
            bar_hover = [f"{get_vn_label_with_unit(bar_c)}=" + (_format_number_vn(v, decimals_auto=False, force_decimals=2) + " %" if _is_pct_bar else _format_number_vn(v)) + f"<br>Năm={int(y)}" for v, y in zip(df_plot[bar_c], df_plot["Year"])]
            fig.add_bar(x=df_plot["Year"], y=df_plot[bar_c], name=get_vn_label_with_unit(bar_c), hovertext=bar_hover, hovertemplate="%{hovertext}<extra></extra>")
            _is_pct_line = "%" in get_vn_label_with_unit(line_c)
            line_hover = [f"{get_vn_label_with_unit(line_c)}=" + (_format_number_vn(v, decimals_auto=False, force_decimals=2) + " %" if _is_pct_line else _format_number_vn(v)) + f"<br>Năm={int(y)}" for v, y in zip(df_plot[line_c], df_plot["Year"])]
            fig.add_trace(go.Scatter(x=df_plot["Year"], y=df_plot[line_c], mode="lines+markers", name=get_vn_label_with_unit(line_c), yaxis="y2", hovertext=line_hover, hovertemplate="%{hovertext}<extra></extra>"))
            fig.update_layout(
                height=450,
                separators=",.",
                xaxis_title="Năm thu thập",
                yaxis=dict(title=get_vn_label_with_unit(bar_c), tickformat=",.2f" if _is_pct_bar else ",.0f"),
                yaxis2=dict(title=get_vn_label_with_unit(line_c), overlaying='y', side='right', tickformat=",.2f" if _is_pct_line else ",.0f"),
                legend_title_text="Chỉ tiêu"
            )
            fig.update_layout(margin=dict(t=60,r=20,b=40,l=60), legend=dict(orientation="h", yanchor="top", y=-0.2), uniformtext_minsize=10, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)

        if "Scatter" in chart_types:
            st.markdown("**Biểu đồ phân tán — Tương quan hai biến**")
            colx, coly = st.columns(2)
            with colx:
                scatter_x = st.selectbox("Chọn X", options=selected_series_for_plot, format_func=lambda c: get_vn_label_with_unit(c), key="scatter_x")
            with coly:
                scatter_y = st.selectbox("Chọn Y", options=[c for c in selected_series_for_plot if c != scatter_x] or selected_series_for_plot,
                                         format_func=lambda c: get_vn_label_with_unit(c), key="scatter_y")
            sc = df_plot[[scatter_x, scatter_y, "Year"]].dropna()
            if sc.empty:
                st.info("Không đủ dữ liệu để vẽ Scatter.")
            else:
                isx = "%" in get_vn_label_with_unit(scatter_x)
                isy = "%" in get_vn_label_with_unit(scatter_y)
                _hover = [f"{get_vn_label_with_unit(scatter_x)}=" + (_format_number_vn(x, decimals_auto=False, force_decimals=2) + " %" if isx else _format_number_vn(x)) + "<br>" + f"{get_vn_label_with_unit(scatter_y)}=" + (_format_number_vn(y, decimals_auto=False, force_decimals=2) + " %" if isy else _format_number_vn(y)) + f"<br>Năm={int(yr)}" for x, y, yr in zip(sc[scatter_x], sc[scatter_y], sc["Year"])]
                fig = px.scatter(sc, x=scatter_x, y=scatter_y, hover_data=None)
                fig.update_traces(hovertext=_hover, hovertemplate="%{hovertext}<extra></extra>")
                fig.update_layout(height=420, separators=",.", xaxis_title="Năm thu thập", yaxis_title=get_vn_label_with_unit(scatter_y))
                fig.update_xaxes(tickformat="d")
                fig.update_yaxes(tickformat=",.2f" if isy else ",.0f")
                trend = add_trendline(sc, scatter_x, scatter_y)
                if trend:
                    x_line, y_line, a, b = trend
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"Đường xu hướng (y≈{a:.2f}x+{b:.2f})"))
                fig.update_layout(margin=dict(t=60,r=20,b=40,l=60), legend=dict(orientation="h", yanchor="top", y=-0.2), uniformtext_minsize=10, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)

        if "Heatmap" in chart_types:
            st.markdown("**Biểu đồ nhiệt — Ma trận tương quan**")
            corr = correlation_matrix(df_plot)
            if corr.empty:
                st.info("Chưa đủ biến số để tính tương quan.")
            else:
                corr_vn = corr.copy()
                corr_vn.columns = [get_vn_label_with_unit(c) for c in corr_vn.columns]
                corr_vn.index = [get_vn_label_with_unit(c) for c in corr_vn.index]
                corr_vn = corr.copy()
                corr_vn.columns = [get_vn_label_with_unit(c) for c in corr_vn.columns]
                corr_vn.index = [get_vn_label_with_unit(c) for c in corr_vn.index]
                _labels = corr.round(2).applymap(lambda v: f"{v:.2f}".replace(".", ","))
                _labels.index = corr_vn.index
                _labels.columns = corr_vn.columns
                fig = go.Figure(data=go.Heatmap(z=corr_vn.values, x=corr_vn.columns, y=corr_vn.index,
                                        colorscale="RdBu", zmid=0, zsmooth=False,
                                        text=_labels.values, texttemplate="%{text}") )
                fig.update_traces(colorbar_tickformat=",.2f")
                fig.update_layout(separators=",.", margin=dict(t=60,r=20,b=40,l=60))
                fig.update_coloraxes(colorbar_tickformat=",.2f")
                fig.update_layout(height=520, coloraxis_colorbar=dict(title="r"))
                fig.update_layout(margin=dict(t=60,r=20,b=40,l=60), legend=dict(orientation="h", yanchor="top", y=-0.2), uniformtext_minsize=10, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)

with tab_stats:
    st.subheader("Bảng thống kê mô tả")
    if stats_df.empty:
        st.info("Chưa có dữ liệu để tính thống kê.")
    else:
        disp = stats_df.copy()
        # Định dạng số theo chuẩn VN
        is_percent_row = disp["Chỉ tiêu"].astype(str).str.contains("%")
        num_cols = ["Giá trị TB (Mean)", "Độ lệch chuẩn (Std)", "Nhỏ nhất (Min)",
                    "Lớn nhất (Max)", "Trung vị (Median)", "Q1", "Q3", "Hệ số biến thiên (CV%)"]
        for c in num_cols:
            if c in disp.columns:
                def _fmt(v, is_pct):
                    if c == "Hệ số biến thiên (CV%)" or is_pct:
                        s = _format_number_vn(v, decimals_auto=False, force_decimals=2)
                        return (s + " %") if s != "" else s
                    return _format_number_vn(v)
                disp[c] = [ _fmt(v, bool(is_percent_row.iloc[i]) if i < len(is_percent_row) else False)
                            for i, v in enumerate(disp[c].tolist()) ]
        disp_show = disp.copy()
        disp_show.index = np.arange(1, len(disp_show) + 1)
        disp_show.index.name = "STT"
        st.dataframe(disp_show, use_container_width=True, height=420)
    st.caption("Nguồn dữ liệu: " + "; ".join(source_list))

with tab_download:
    st.subheader("Tải dữ liệu")
    if imputed_df.empty:
        st.info("Chưa có dữ liệu để tải.")
    else:
        bytes_xlsx = to_excel_bytes(imputed_df, stats_df, corr_df if not corr_df.empty else pd.DataFrame())
        st.download_button(
            label="⬇️ Tải Excel (Data + Stats + Correlation)",
            data=bytes_xlsx,
            file_name=f"macro_{sel_country}_{start_year}-{end_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        csv_bytes = imputed_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Tải CSV (Data)",
            data=csv_bytes,
            file_name=f"macro_{sel_country}_{start_year}-{end_year}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Nguồn dữ liệu: " + "; ".join(source_list))

with tab_ai:
    # (2) KHÔNG hiển thị C-ADAPT; đề mục tiếng Việt
    st.subheader("AI phân tích và tư vấn")
    audience = st.selectbox("Đối tượng tư vấn", ["Nhà đầu tư", "Doanh nghiệp", "Ngân hàng (Agribank)"])

    def build_ai_prompt(audience: str, country_label: str, year_range: str,
                        stats_df: pd.DataFrame, corr_df: pd.DataFrame, selected_cols: list) -> str:
        # Tóm tắt biến động: top CV%
        top_lines = []
        if not stats_df.empty and "Hệ số biến thiên (CV%)" in stats_df.columns:
            tmp_cv = stats_df.sort_values("Hệ số biến thiên (CV%)", ascending=False).head(3)
            for _, r in tmp_cv.iterrows():
                cvv = r["Hệ số biến thiên (CV%)"] if pd.notna(r["Hệ số biến thiên (CV%)"]) else 0
                minv = r["Nhỏ nhất (Min)"] if pd.notna(r["Nhỏ nhất (Min)"]) else 0
                maxv = r["Lớn nhất (Max)"] if pd.notna(r["Lớn nhất (Max)"]) else 0
                top_lines.append(f"- {r['Chỉ tiêu']}: CV≈{cvv:.1f}%, Min {minv:.2f} ({r['Năm Min']}), Max {maxv:.2f} ({r['Năm Max']}).")

        corr_lines = []
        if not corr_df.empty:
            cols = list(corr_df.columns)
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    r = corr_df.iloc[i, j]
                    if pd.notna(r) and abs(r) >= 0.7:
                        corr_lines.append(f"- {get_vn_label_with_unit(cols[i])} vs {get_vn_label_with_unit(cols[j])}: r={float(r):.2f}")
        if not corr_lines:
            corr_lines = ["- Chưa có tương quan mạnh (|r| ≥ 0.7)."]

        # Chỉ hiển thị khuyến nghị cho đúng đối tượng
        if audience == "Nhà đầu tư":
            advice_block = """- Phân bổ (CK/BĐS/vàng/FX) theo 2–3 kịch bản (cơ sở, tích cực, thận trọng) với ngưỡng kích hoạt.
- Quản trị rủi ro: stop-loss, tái cân bằng theo biến động lạm phát/lãi suất."""
        elif audience == "Doanh nghiệp":
            advice_block = """- Kế hoạch sản xuất/vốn/XNK theo kịch bản cầu nội địa & tỷ giá.
- Quản trị rủi ro chi phí vốn (lãi suất) và tỷ giá; tối ưu tồn kho, chu kỳ tiền mặt."""
        else:
            advice_block = """- Gói cho vay ưu đãi theo ngành ưu tiên; linh hoạt kỳ hạn/lãi suất theo kịch bản.
- Tiêu chí thẩm định: DSCR, vòng quay vốn, độ nhạy lãi suất/tỷ giá; xếp hạng tín dụng nội bộ."""

        # Prompt hoàn toàn tiếng Việt, không nhắc C-ADAPT
        prompt = f"""
Bạn là chuyên gia kinh tế & tài chính. Hãy phân tích dữ liệu vĩ mô của {country_label} giai đoạn {year_range}.
Trình bày NGẮN GỌN theo các đề mục sau (chỉ dùng tiêu đề tiếng Việt):

1) Bối cảnh & Dữ liệu chính:
- Chỉ tiêu đang phân tích: {', '.join([get_vn_label_with_unit(c) for c in selected_cols])}.

2) Xu hướng nổi bật & Biến động:
{os.linesep.join(top_lines) if top_lines else "- (Dữ liệu hạn chế để tóm tắt chi tiết)"} 

3) Tương quan đáng chú ý:
{os.linesep.join(corr_lines)}

4) Kiến nghị cho đối tượng: {audience}
{advice_block}

5) Hành động thực thi (kèm KPI/điều kiện kích hoạt):

6) Rủi ro chính & Cách kiểm chứng sau mỗi kỳ công bố dữ liệu:
"""
        return prompt.strip()

    if st.button("🚀 Sinh AI phân tích và tư vấn"):
        if imputed_df.empty:
            st.info("Chưa có dữ liệu để phân tích.")
        else:
            prompt = build_ai_prompt(
                audience=audience,
                country_label=sel_country_name,
                year_range=f"{start_year}-{end_year}",
                stats_df=stats_df,
                corr_df=corr_df,
                selected_cols=[c for c in imputed_df.columns if c != "Year"]
            )
            if not OPENAI_OK:
                st.warning("⚠️ Mô-đun AI chưa sẵn sàng (thiếu thư viện openai). Hãy cài đặt và cấu hình OPENAI_API_KEY.")
            else:
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    st.warning("⚠️ Chưa phát hiện OPENAI_API_KEY. Vui lòng đặt biến môi trường.")
                else:
                    try:
                        client = OpenAI(api_key=api_key)
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "Bạn là chuyên gia kinh tế vĩ mô & tài chính, viết ngắn gọn, súc tích, dùng tiêu đề tiếng Việt."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.4,
                            max_tokens=900,
                        )
                        st.markdown(resp.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Lỗi khi gọi OpenAI: {e}")

# Footer
