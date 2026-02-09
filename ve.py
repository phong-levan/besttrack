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
from datetime import datetime
import pytz

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
LINK_KMA_FORECAST = "https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136"

SIDEBAR_WIDTH = "320px"
COLOR_SIDEBAR = "#f8f9fa"

st.set_page_config(page_title="Hệ thống giám sát", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# 2. CSS TỐI ƯU (CĂN GIỮA BẢNG & BỎ THANH CUỘN)
# ==============================================================================
st.markdown(f"""
    <style>
    /* Triệt tiêu thanh cuộn toàn trang */
    html, body, [data-testid="stAppViewContainer"] {{
        overflow: hidden !important;
        height: 100vh !important;
    }}

    /* Loại bỏ khoảng trống thừa của Streamlit */
    .block-container {{
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }}
    
    header, footer {{ display: none !important; }}

    /* Fix cứng Sidebar trái và bỏ thanh cuộn sidebar */
    section[data-testid="stSidebar"] {{
        width: {SIDEBAR_WIDTH} !important;
        background-color: {COLOR_SIDEBAR} !important;
        border-right: 1px solid #ddd;
        overflow-x: hidden !important;
        overflow-y: hidden !important;
    }}

    /* Widget nổi bên phải */
    .legend-box {{
        position: fixed; top: 10px; right: 20px; z-index: 9999;
        width: 280px; pointer-events: none;
    }}

    .info-box {{
        position: fixed; top: 220px; right: 20px; z-index: 9999;
        width: 340px; background: rgba(255, 255, 255, 0.95);
        border: 1px solid #ccc; border-radius: 8px;
        padding: 10px !important; color: #000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}

    .info-title {{
        text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 2px; color: #d32f2f;
    }}

    .info-subtitle {{
        text-align: center; font-size: 11px; margin-bottom: 8px; font-style: italic; color: #555;
    }}

    /* CĂN GIỮA BẢNG TRONG INFO-BOX */
    .info-box table {{
        margin-left: auto;
        margin-right: auto;
        border-collapse: collapse;
        width: 100%;
        font-size: 12px;
    }}

    .info-box th, .info-box td {{
        text-align: center !important;
        padding: 5px 2px;
        border-bottom: 1px solid #eee;
    }}

    /* Đẩy map sang phải để không bị sidebar che */
    [data-testid="stAppViewContainer"] {{
        padding-left: {SIDEBAR_WIDTH} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HÀM XỬ LÝ
# ==============================================================================

def image_to_base64(image_path):
    if not os.path.exists(image_path): return None
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {
        "tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no",
        "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "khí áp": "pressure", "pmin": "pressure"
    }
    return df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

def create_info_table(df, title):
    # Lấy giờ VN hiện tại để tránh lỗi NameError
    tz = pytz.timezone('Asia/Ho_Chi_Minh')
    current_time_str = datetime.now(tz).strftime("%H:%M %d/%m/%Y")
    
    if df.empty:
        return f'<div class="info-box"><div class="info-title">{title}</div><div class="info-subtitle">Đang đợi dữ liệu...</div></div>'
    
    # Lấy 8 bản tin mới nhất
    display_df = df.tail(8)
    rows = ""
    for _, r in display_df.iterrows():
        t = r.get('datetime_str', '-')
        lon, lat = f"{r.get('lon', 0):.1f}E", f"{r.get('lat', 0):.1f}N"
        bf = r.get('bf', 0)
        cap = f"Cấp {int(bf)}" if bf > 0 else "-"
        pmin = f"{int(r.get('pressure', 0))}" if r.get('pressure', 0) > 0 else "-"
        rows += f"<tr><td>{t}</td><td>{lon}</td><td>{lat}</td><td>{cap}</td><td>{pmin}</td></tr>"

    return textwrap.dedent(f"""
        <div class="info-box">
            <div class="info-title">{title}</div>
            <div class="info-subtitle">Cập nhật: {current_time_str}</div>
            <table>
                <thead>
                    <tr><th>Ngày-Giờ</th><th>Kinh độ</th><th>Vĩ độ</th><th>Gió</th><th>Pmin</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    """)

# ... (Các hàm densify_track, generate_circle_polygon, create_storm_swaths, get_icon_name giữ nguyên như bản trước) ...
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
            new_rows.append(row)
    new_rows.append(df.iloc[-1])
    return pd.DataFrame(new_rows)

def generate_circle_polygon(lat, lon, radius_km, n_points=36):
    if radius_km <= 0: return None
    coords = []
    for i in range(n_points):
        theta = (i / n_points) * (2 * pi)
        dy = (radius_km * cos(theta)) / 111.32
        dx = (radius_km * sin(theta)) / (111.32 * cos(radians(lat)))
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
    bf = row.get('bf', 0)
    status = 'daqua' if any(x in str(row.get('status_raw','')).lower() for x in ['quá khứ', 'past']) else 'dubao'
    if bf < 6: return f"vungthap_{status}"
    if bf < 8: return f"atnd_{status}"
    if bf <= 11: return f"bnd_{status}"
    return f"sieubao_{status}"

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    with st.sidebar:
        st.title("HỆ THỐNG GIÁM SÁT")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = "THÔNG TIN BÃO"
        show_widgets = False

        if topic == "Bản đồ Bão":
            storm_opt = st.selectbox("Dữ liệu:", ["Hiện trạng (Besttrack)", "Lịch sử"])
            show_widgets = st.checkbox("Hiển thị bảng tin", value=True)
            
            f = st.file_uploader("Tải file dữ liệu:", type=["csv", "xlsx"])
            path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
            
            if path:
                try:
                    df = pd.read_csv(path) if str(path).endswith('.csv') or (f and f.name.endswith('.csv')) else pd.read_excel(path)
                    df = normalize_columns(df)
                    if 'storm_no' in df.columns:
                        all_s = df['storm_no'].unique()
                        sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s[:1])
                        final_df = df[df['storm_no'].isin(sel)]
                except: st.error("Lỗi định dạng file.")

    # Main Content Area
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&zoom=5&overlay=satellite", height=1000)
    elif topic == "Dữ liệu quan trắc":
        components.iframe(LINK_WEATHEROBS, height=1000, scrolling=True)
    elif topic == "Dự báo điểm (KMA)":
        components.iframe(LINK_KMA_FORECAST, height=1000, scrolling=True)
    elif topic == "Bản đồ Bão":
        # Khởi tạo bản đồ
        m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles='CartoDB positron', zoom_control=False)
        folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                         attr='Esri', name='Vệ tinh').add_to(m)

        if not final_df.empty:
            fg = folium.FeatureGroup(name="Dữ liệu bão")
            dense = densify_track(final_df)
            f6, f10, fc = create_storm_swaths(dense)
            
            # Vẽ vùng ảnh hưởng
            for geom, c, o in [(f6,'#FFC0CB',0.3), (f10,'#FF6347',0.4), (fc,'#90EE90',0.5)]:
                if geom and not geom.is_empty:
                    folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg)
            
            # Đường đi
            folium.PolyLine(final_df[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg)
            
            # Icon
            for _, r in final_df.iterrows():
                ik = get_icon_name(r)
                ib64 = image_to_base64(ICON_PATHS.get(ik, ""))
                if ib64:
                    folium.Marker([r['lat'], r['lon']], icon=folium.CustomIcon(ib64, icon_size=(36, 36))).add_to(fg)
            fg.add_to(m)

        folium.LayerControl(position='topleft').add_to(m)

        # Hiển thị Widget nổi
        if show_widgets:
            st.markdown(create_info_table(final_df, dashboard_title), unsafe_allow_html=True)
            img_b64 = image_to_base64(CHUTHICH_IMG)
            if img_b64:
                st.markdown(f'<div class="legend-box"><img src="{img_b64}" style="width:100%"></div>', unsafe_allow_html=True)

        st_folium(m, width=2000, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
