# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
import requests
import streamlit.components.v1 as components
from math import radians, sin, cos, asin, sqrt
import warnings
import textwrap

# Thư viện hình học
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"
FILE_OPT2 = "besttrack_capgio.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# --- DANH SÁCH LINK WEB ---
LINK_WEATHEROBS = "https://weatherobs.com/"
LINK_WIND_AUTO = "https://kttvtudong.net/kttv"
LINK_KMA_FORECAST = "https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm=2026.02.06.12&delta=000&ftm=2026.02.06.12"

# Màu sắc
COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
COLOR_TEXT = "#333333"
COLOR_ACCENT = "#007bff"
COLOR_BORDER = "#dee2e6"
SIDEBAR_WIDTH = "320px"

st.set_page_config(
    page_title="Storm Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CSS CHUNG (FIX CỨNG & STYLE MỚI)
# ==============================================================================
st.markdown(f"""
    <style>
    /* 1. KHÓA CUỘN TRANG CHÍNH */
    html, body, .stApp {{
        overflow: hidden !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        font-family: Arial, sans-serif !important;
    }}

    /* 2. ẨN HEADER & FOOTER */
    header, footer, [data-testid="stHeader"], [data-testid="stToolbar"] {{
        display: none !important;
    }}
    .block-container {{
        padding: 0 !important; margin: 0 !important; max-width: 100vw !important;
    }}
    
    /* 3. SIDEBAR (CỐ ĐỊNH) */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_SIDEBAR} !important;
        border-right: 1px solid {COLOR_BORDER};
        width: {SIDEBAR_WIDTH} !important;
        min-width: {SIDEBAR_WIDTH} !important;
        max-width: {SIDEBAR_WIDTH} !important;
        top: 0 !important;
        height: 100vh !important;
        z-index: 9999999 !important;
        position: fixed !important;
        left: 0 !important;
        padding-top: 0 !important;
    }}
    
    [data-testid="stSidebarUserContent"] {{
        padding: 20px;
        height: 100vh;
        overflow-y: auto !important;
    }}
    
    [data-testid="stSidebarCollapseBtn"] {{ display: none !important; }}
    
    [data-testid="stSidebarCollapsedControl"] {{
        display: flex !important; z-index: 1000000;
        top: 10px; left: 10px; background: white; border: 1px solid #ccc;
    }}

    /* 4. FULL SCREEN MAP/IFRAME */
    iframe, [data-testid="stFoliumMap"] {{
        position: fixed !important;
        top: 0 !important;
        left: {SIDEBAR_WIDTH} !important;
        width: calc(100vw - {SIDEBAR_WIDTH}) !important;
        height: 100vh !important;
        border: none !important;
        z-index: 1 !important;
        display: block !important;
    }}

    /* -----------------------------------------------------------
       5. STYLE MỚI CHO BẢNG CHÚ THÍCH (LEGEND) - CHỈ CÓ ẢNH
    ----------------------------------------------------------- */
    .legend-box {{
        position: fixed; 
        top: 20px; 
        right: 20px; 
        z-index: 10000;
        width: 300px; 
        /* Xóa bỏ hoàn toàn khung viền và nền */
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }}
    .legend-box img {{
        width: 100%;
        display: block;
        /* Nếu ảnh gốc không có viền trắng, có thể thêm bo góc nhẹ nếu muốn */
        border-radius: 4px; 
    }}

    /* -----------------------------------------------------------
       6. STYLE MỚI CHO BẢNG THÔNG TIN (INFO TABLE) - DẠNG VĂN BẢN
    ----------------------------------------------------------- */
    .info-box {{
        position: fixed; 
        top: 250px; /* Cách top đủ xa để nằm dưới chú thích */
        right: 20px; 
        z-index: 9999;
        width: 450px; /* Rộng hơn để chứa đủ cột */
        
        /* Style Bảng Trắng */
        background: rgba(255, 255, 255, 0.95); /* Trắng đục */
        border: 1px solid #ccc; /* Viền mảnh màu xám */
        border-radius: 0px; /* Vuông vức, không bo tròn nhiều */
        color: #000; /* Chữ đen */
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    
    /* Tiêu đề bảng */
    .info-title {{
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 5px;
        text-transform: uppercase;
        color: #000;
    }}
    
    .info-subtitle {{
        text-align: center;
        font-size: 13px;
        margin-bottom: 10px;
        font-style: italic;
        color: #333;
    }}

    /* Cấu trúc Table bên trong */
    table {{ 
        width: 100%; 
        border-collapse: collapse; 
        font-size: 14px; 
        color: #000;
    }}
    
    th {{ 
        background-color: transparent !important; /* Bỏ màu nền header */
        color: #000 !important; /* Chữ đen */
        padding: 8px; 
        font-weight: bold; 
        border-bottom: 2px solid #000; /* Đường kẻ đậm dưới header */
        text-align: center;
    }}
    
    td {{ 
        padding: 6px; 
        border-bottom: 1px solid #ccc; /* Đường kẻ mờ giữa các dòng */
        text-align: center; 
        color: #000;
    }}
    
    /* Layer Control mặc định của Folium */
    .leaflet-control-layers {{
        background: white !important; color: #333 !important;
        border: 1px solid #ccc !important; padding: 10px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HÀM XỬ LÝ LOGIC
# ==============================================================================

@st.cache_data(ttl=300) 
def get_rainviewer_ts():
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        r = requests.get(url, timeout=3, verify=False)
        return r.json()['satellite']['infrared'][-1]['time']
    except: return None

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {
        "tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no",
        "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "vĩ độ": "lat", "kinh độ": "lon", "gió (kt)": "wind_kt",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "khí áp": "pressure", "khí áp (mb)": "pressure", "pmin": "pressure"
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    return df

def densify_track(df, step_km=10):
    new_rows = []
    if len(df) < 2: return df
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = 6371 * 2 * asin(sqrt(sin(radians(p2['lat']-p1['lat'])/2)**2 + cos(radians(p1['lat']))*cos(radians(p2['lat']))*sin(radians(p2['lon']-p1['lon'])/2)**2))
        steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(steps):
            f = j / steps
            row = p1.copy()
            row['lat'] = p1['lat'] + (p2['lat'] - p1['lat']) * f
            row['lon'] = p1['lon'] + (p2['lon'] - p1['lon']) * f
            for col in ['r6', 'r10', 'rc']:
                if col in p1 and col in p2: row[col] = p1.get(col, 0)*(1-f) + p2.get(col, 0)*f
            new_rows.append(row)
    new_rows.append(df.iloc[-1])
    return pd.DataFrame(new_rows)

def create_storm_swaths(dense_df):
    polys = {'r6': [], 'r10': [], 'rc': []}
    geo = geodesic.Geodesic()
    for _, row in dense_df.iterrows():
        for r, key in [(row.get('r6',0), 'r6'), (row.get('r10',0), 'r10'), (row.get('rc',0), 'rc')]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                polys[key].append(Polygon(circle))
    u = {k: unary_union(v) if v else None for k, v in polys.items()}
    f_rc = u['rc']
    f_r10 = u['r10'].difference(u['rc']) if u['r10'] and u['rc'] else u['r10']
    f_r6 = u['r6'].difference(u['r10']) if u['r6'] and u['r10'] else u['r6']
    return f_r6, f_r10, f_rc

def get_icon_name(row):
    w = row.get('wind_kt', 0)
    bf = row.get('bf', 0)
    if pd.isna(bf) or bf == 0:
        if w < 34: bf = 6
        elif w < 64: bf = 8
        elif w < 100: bf = 10
        else: bf = 12
    status = 'dubao' if 'forecast' in str(row.get('status_raw','')).lower() else 'daqua'
    if bf < 6: return f"vungthap_{status}"
    if bf < 8: return f"atnd_{status}"
    if bf <= 11: return f"bnd_{status}"
    return f"sieubao_{status}"

# === HÀM TẠO BẢNG THÔNG TIN (STYLE TRẮNG ĐEN) ===
def create_info_table(df, title):
    if df.empty: return ""
    if 'status_raw' in df.columns:
         cur = df[df['status_raw'].astype(str).str.contains("hiện tại|current", case=False, na=False)]
         fut = df[df['status_raw'].astype(str).str.contains("dự báo|forecast", case=False, na=False)]
         display_df = pd.concat([cur, fut]).head(8)
    else:
         display_df = df.sort_values('dt', ascending=False).groupby('name').head(1)

    rows = ""
    for _, r in display_df.iterrows():
        t = r.get('datetime_str', r.get('dt'))
        if not isinstance(t, str): t = t.strftime('%d/%m %Hh')
        
        lon = f"{r.get('lon', 0):.1f}E"
        lat = f"{r.get('lat', 0):.1f}N"
        
        bf = r.get('bf', 0)
        w = r.get('wind_kt', 0)
        if (pd.isna(bf) or bf == 0) and w > 0:
             if w < 34: bf = 6
             elif w < 64: bf = 8
             elif w < 100: bf = 10
             else: bf = 12
        cap_gio = f"Cấp {int(bf)}" if bf > 0 else "-"
        
        p = r.get('pressure', 0)
        pmin = f"{int(p)}" if (pd.notna(p) and p > 0) else "-"

        rows += f"<tr><td>{t}</td><td>{lon}</td><td>{lat}</td><td>{cap_gio}</td><td>{pmin}</td></tr>"
    
    # Bảng trắng tinh, không màu mè
    return textwrap.dedent(f"""
    <div class="info-box">
        <div class="info-title">{title}</div>
        <div class="info-subtitle">(Dữ liệu cập nhật từ Besttrack)</div>
        <table>
            <thead>
                <tr>
                    <th>Ngày - Giờ</th>
                    <th>Kinh độ</th>
                    <th>Vĩ độ</th>
                    <th>Cấp gió</th>
                    <th>Pmin(mb)</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""")

# === HÀM TẠO CHÚ GIẢI (CHỈ ẢNH, KHÔNG KHUNG) ===
def create_legend(img_b64):
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="legend-box">
        <img src="data:image/png;base64,{img_b64}">
    </div>""")

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    
    with st.sidebar:
