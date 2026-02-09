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

# --- DANH SÁCH LINK WEB ---
LINK_WEATHEROBS = "https://weatherobs.com/"
LINK_WIND_AUTO = "https://kttvtudong.net/kttv"
# Cập nhật link KMA với thời gian động nếu cần, ở đây giữ nguyên mẫu của bạn
LINK_KMA_FORECAST = "https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136"

# Màu sắc
COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
SIDEBAR_WIDTH = "320px"

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống giám sát bão",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# ==============================================================================
# 2. CSS CHUNG
# ==============================================================================
st.markdown(f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        overflow: hidden !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    .block-container {{
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        height: 100vh !important;
    }}
    header, footer {{ display: none !important; }}
    section[data-testid="stSidebar"] {{
        width: {SIDEBAR_WIDTH} !important;
        background-color: {COLOR_SIDEBAR} !important;
        border-right: 1px solid #ddd;
    }}
    .info-box {{
        position: fixed; top: 220px; right: 20px; z-index: 9999;
        width: 320px; background: rgba(255, 255, 255, 0.95);
        border: 1px solid #ccc; border-radius: 8px; padding: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }}
    .info-title {{ text-align: center; font-weight: bold; font-size: 16px; color: red; }}
    .info-subtitle {{ text-align: center; font-size: 11px; font-style: italic; margin-bottom: 10px; }}
    .info-box table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .info-box th, .info-box td {{ border: 1px solid #eee; padding: 4px; text-align: center; }}
    .legend-box {{ position: fixed; top: 10px; right: 20px; z-index: 9999; width: 280px; }}
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
        "khí áp": "pressure", "khí áp (mb)": "pressure", "pmin": "pressure"
    }
    return df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

def densify_track(df, step_km=10):
    if len(df) < 2: return df
    new_rows = []
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
        for r_val, key in [(row.get('r6',0), 'r6'), (row.get('r10',0), 'r10'), (row.get('rc',0), 'rc')]:
            if r_val > 0:
                poly = generate_circle_polygon(row['lat'], row['lon'], r_val)
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

def create_info_table(df, title):
    # FIX LỖI NameError tại đây
    tz = pytz.timezone('Asia/Ho_Chi_Minh')
    current_time_str = datetime.now(tz).strftime("%H:%M %d/%m/%Y")
    
    if df.empty: return f'<div class="info-box"><div class="info-title">{title}</div><div class="info-subtitle">Không có dữ liệu</div></div>'
    
    # Lấy tối đa 8 dòng dữ liệu mới nhất
    display_df = df.tail(8)
    rows = ""
    for _, r in display_df.iterrows():
        t = r.get('datetime_str', '-')
        lon, lat = f"{r['lon']:.1f}E", f"{r['lat']:.1f}N"
        cap = f"Cấp {int(r['bf'])}" if r['bf'] > 0 else "-"
        pmin = f"{int(r['pressure'])}" if r['pressure'] > 0 else "-"
        rows += f"<tr><td>{t}</td><td>{lon}</td><td>{lat}</td><td>{cap}</td><td>{pmin}</td></tr>"

    return textwrap.dedent(f"""
        <div class="info-box">
            <div class="info-title">{title}</div>
            <div class="info-subtitle">Cập nhật lúc: {current_time_str}</div>
            <table>
                <thead><tr><th>Ngày-Giờ</th><th>Kinh độ</th><th>Vĩ độ</th><th>Gió</th><th>Pmin</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    """)

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    with st.sidebar:
        st.title("GIÁM SÁT THIÊN TAI")
        topic = st.radio("CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = "DỮ LIỆU"
        show_widgets = False

        if topic == "Bản đồ Bão":
            mode = st.selectbox("Nguồn dữ liệu:", ["Hiện trạng (Besttrack)", "Lịch sử"])
            show_widgets = st.checkbox("Hiển thị bảng thông tin", value=True)
            file_path = FILE_OPT1 if "Hiện trạng" in mode else FILE_OPT2
            
            uploaded_file = st.file_uploader("Tải file mới (tùy chọn):", type=["csv", "xlsx"])
            target = uploaded_file if uploaded_file else (file_path if os.path.exists(file_path) else None)

            if target:
                try:
                    df = pd.read_csv(target) if str(target).endswith('.csv') or hasattr(target, 'name') and target.name.endswith('.csv') else pd.read_excel(target)
                    df = normalize_columns(df)
                    if 'storm_no' in df.columns:
                        ids = df['storm_no'].unique()
                        sel = st.multiselect("Chọn cơn bão:", ids, default=ids[:1])
                        final_df = df[df['storm_no'].isin(sel)]
                        dashboard_title = f"BÃO: {', '.join(map(str, sel))}"
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")

    # --- RENDER NỘI DUNG ---
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&zoom=5&overlay=satellite", height=1000)
    elif topic == "Dữ liệu quan trắc":
        components.iframe(LINK_WEATHEROBS, height=1000, scrolling=True)
    elif topic == "Dự báo điểm (KMA)":
        components.iframe(LINK_KMA_FORECAST, height=1000, scrolling=True)
    elif topic == "Bản đồ Bão":
        m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles='CartoDB positron')
        
        # Lớp vệ tinh
        folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                         attr='Esri', name='Vệ tinh').add_to(m)
        
        # Lớp Radar/Mây từ RainViewer
        ts = get_rainviewer_ts()
        if ts:
            folium.TileLayer(tiles=f"https://tile.rainviewer.com/{ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png", 
                             attr="RainViewer", name="Mây vệ tinh (Live)", overlay=True, opacity=0.5).add_to(m)

        if not final_df.empty:
            fg = folium.FeatureGroup(name="Hành trình bão")
            # Vẽ đường đi & vùng ảnh hưởng
            dense = densify_track(final_df)
            f6, f10, fc = create_storm_swaths(dense)
            
            for geom, color, op in [(f6, '#FFC0CB', 0.3), (f10, '#FF6347', 0.4), (fc, '#90EE90', 0.5)]:
                if geom and not geom.is_empty:
                    folium.GeoJson(mapping(geom), style_function=lambda x,c=color,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg)
            
            folium.PolyLine(final_df[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg)
            
            # Marker Icons
            for _, r in final_df.iterrows():
                icon_k = get_icon_name(r)
                icon_b64 = image_to_base64(ICON_PATHS.get(icon_k, ""))
                if icon_b64:
                    folium.Marker([r['lat'], r['lon']], 
                                  icon=folium.CustomIcon(icon_b64, icon_size=(34, 34)),
                                  tooltip=f"Gió: {r['wind_km/h']} km/h").add_to(fg)
            fg.add_to(m)

        folium.LayerControl().add_to(m)
        
        # Widgets nổi
        if show_widgets:
            st.markdown(create_info_table(final_df, dashboard_title), unsafe_allow_html=True)
            if os.path.exists(CHUTHICH_IMG):
                st.markdown(f'<div class="legend-box"><img src="{image_to_base64(CHUTHICH_IMG)}" style="width:100%"></div>', unsafe_allow_html=True)

        st_folium(m, width=2000, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
