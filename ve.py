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
from math import radians, sin, cos, asin, sqrt, pi
import warnings
import textwrap

# Thư viện hình học
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.csv"
FILE_OPT2 = "besttrack_capgio.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# --- ĐỊNH NGHĨA ICON PATHS ---
ICON_PATHS = {
    "vungthap_daqua": os.path.join(ICON_DIR, 'vungthapdaqua.png'),
    "atnd_daqua": os.path.join(ICON_DIR, 'atnddaqua.PNG'),
    "bnd_daqua": os.path.join(ICON_DIR, 'bnddaqua.PNG'),
    "sieubao_daqua": os.path.join(ICON_DIR, 'sieubaodaqua.PNG'),
    "vungthap_dubao": os.path.join(ICON_DIR, 'vungthapdubao.png'),
    "atnd_dubao": os.path.join(ICON_DIR, 'atnd.PNG'),
    "bnd_dubao": os.path.join(ICON_DIR, 'bnd.PNG'),
    "sieubao_dubao": os.path.join(ICON_DIR, 'sieubao.PNG')
}

LINK_WEATHEROBS = "https://weatherobs.com/"
LINK_WIND_AUTO = "https://kttvtudong.net/kttv"
LINK_KMA_FORECAST = "https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm=2026.02.06.12&delta=000&ftm=2026.02.06.12"

COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
COLOR_TEXT = "#333333"
COLOR_ACCENT = "#007bff"
COLOR_BORDER = "#dee2e6"

# CẤU HÌNH TRANG
st.set_page_config(
    page_title="Storm Monitor",
    layout="wide",
    initial_sidebar_state="expanded" # Cố gắng mở rộng
)

# ==============================================================================
# 2. CSS CHUNG (ĐÃ RESET ĐỂ HIỆN LẠI NÚT MENU)
# ==============================================================================
st.markdown(f"""
    <style>
    /* 1. Reset lề */
    .block-container {{
        padding: 0 !important; margin: 0 !important; max-width: 100vw !important;
    }}
    
    /* Ẩn Header mặc định */
    header[data-testid="stHeader"] {{ display: none !important; }}
    footer {{ display: none !important; }}

    /* 2. KHÔI PHỤC NÚT MỞ MENU (QUAN TRỌNG NHẤT) */
    /* Đây là nút mũi tên > hiện ra khi menu bị đóng */
    [data-testid="stSidebarCollapsedControl"] {{
        display: block !important; /* BẮT BUỘC HIỆN */
        color: #000 !important;
        background-color: white !important;
        border: 2px solid #007bff; /* Viền xanh để dễ nhìn */
        border-radius: 5px;
        top: 15px !important;
        left: 15px !important;
        z-index: 1000000 !important;
        width: 40px;
        height: 40px;
    }}

    /* 3. TÙY CHỈNH THANH SIDEBAR */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_SIDEBAR} !important;
        border-right: 1px solid #ddd;
    }}
    
    /* Ẩn nút "Thu gọn" bên trong menu để tránh bấm nhầm lần nữa */
    [data-testid="stSidebarCollapseBtn"] {{
        display: none !important;
    }}

    /* 4. BẢN ĐỒ */
    iframe {{
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
    }}
    
    /* 5. CÁC WIDGET NỔI */
    .legend-box {{
        position: fixed; top: 20px; right: 20px; z-index: 999;
        width: 300px; pointer-events: none; 
    }}
    .legend-box img {{ width: 100%; display: block; }}

    .info-box {{
        position: fixed; top: 250px; right: 20px; z-index: 999;
        width: fit-content !important; min-width: 150px; 
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #ccc;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        padding: 5px !important; color: #000; border-radius: 6px;
    }}
    
    .info-title {{
        text-align: center; font-weight: bold; font-size: 16px; margin: 0 0 5px 0; 
        text-transform: uppercase; color: #000;
    }}
    .info-subtitle {{
        text-align: center; font-size: 11px; margin-bottom: 5px; font-style: italic; color: #333;
    }}
    table {{ 
        border-collapse: collapse; font-size: 13px; color: #000; white-space: nowrap; margin: 0;
    }}
    th {{ 
        background: transparent !important; color: #000 !important; padding: 4px 8px; 
        font-weight: bold; border-bottom: 1px solid #000; text-align: center;
    }}
    td {{ 
        padding: 4px 8px; border-bottom: 1px solid #ccc; text-align: center; color: #000; 
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

def image_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    ext = image_path.split('.')[-1].lower()
    mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {
        "tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no",
        "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "khí áp": "pressure", "khí áp (mb)": "pressure", "pmin": "pressure", "pmin (mb)": "pressure"
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

def generate_circle_polygon(lat, lon, radius_km, n_points=36):
    coords = []
    if radius_km <= 0: return None
    lat_rad = radians(lat)
    for i in range(n_points):
        theta = (i / n_points) * (2 * pi)
        dy = (radius_km * cos(theta)) / 111.32
        dx = (radius_km * sin(theta)) / (111.32 * cos(lat_rad))
        coords.append((lon + dx, lat + dy))
    return Polygon(coords)

def create_storm_swaths(dense_df):
    polys = {'r6': [], 'r10': [], 'rc': []}
    for _, row in dense_df.iterrows():
        for r, key in [(row.get('r6',0), 'r6'), (row.get('r10',0), 'r10'), (row.get('rc',0), 'rc')]:
            if r > 0:
                poly = generate_circle_polygon(row['lat'], row['lon'], r)
                if poly: polys[key].append(poly)
    u = {k: unary_union(v) if v else None for k, v in polys.items()}
    f_rc = u['rc']
    f_r10 = u['r10'].difference(u['rc']) if u['r10'] and u['rc'] else u['r10']
    f_r6 = u['r6'].difference(u['r10']) if u['r6'] and u['r10'] else u['r6']
    return f_r6, f_r10, f_rc

def get_icon_name(row):
    wind_speed = row.get('bf', 0) 
    w = row.get('wind_km/h', 0)
    
    if pd.isna(wind_speed) or wind_speed == 0:
        if w > 0:
            if w < 34: wind_speed = 5
            elif w < 64: wind_speed = 7
            elif w < 100: wind_speed = 10
            else: wind_speed = 12
    
    status_raw = str(row.get('status_raw','')).lower()
    
    status = 'dubao' 
    if 'quá khứ' in status_raw or 'past' in status_raw:
        status = 'daqua'
    
    if pd.isna(wind_speed): return f"vungthap_{status}"
    if wind_speed < 6:      return f"vungthap_{status}"
    if wind_speed < 8:      return f"atnd_{status}"
    if wind_speed <= 11:    return f"bnd_{status}"
    return f"sieubao_{status}"

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
        w = r.get('wind_km/h', 0)
        
        lon = f"{r.get('lon', 0):.1f}E"
        lat = f"{r.get('lat', 0):.1f}N"
        
        bf = r.get('bf', 0)
        if (pd.isna(bf) or bf == 0) and w > 0:
             if w < 34: bf = 6
             elif w < 64: bf = 8
             elif w < 100: bf = 10
             else: bf = 12
        cap_gio = f"Cấp {int(bf)}" if bf > 0 else "-"
        
        p = r.get('pressure', 0)
        pmin = f"{int(p)}" if (pd.notna(p) and p > 0) else "-"

        rows += f"<tr><td>{t}</td><td>{lon}</td><td>{lat}</td><td>{cap_gio}</td><td>{pmin}</td></tr>"
    
    return textwrap.dedent(f"""
    <div class="info-box">
        <div class="info-title">{title}</div>
        <div class="info-subtitle">(Dữ liệu cập nhật từ Besttrack)</div>
        <table>
            <thead>
                <tr>
                    <th>Ngày-Giờ</th>
                    <th>Kinh độ</th>
                    <th>Vĩ độ</th>
                    <th>Cấp gió</th>
                    <th>Pmin</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""")

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
        st.title("🌪️ TRUNG TÂM BÃO")
        
        topic = st.radio("CHỌN CHẾ ĐỘ:", 
                         ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""
        obs_mode = ""

        if topic == "Dữ liệu quan trắc":
            obs_mode =
