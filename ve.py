# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic
from folium.plugins import SimpleScreenshot, MousePosition, MeasureControl

# --- 1. CẤU HÌNH HỆ THỐNG (Dựa trên file làm web.docx) ---
DATA_FOLDER = "besttrack"  
ICON_DIR = "icon"
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(page_title="Hệ thống Theo dõi Bão - Modular Version", layout="wide")

# CSS: GIỮ NGUYÊN CODE NỀN (Full màn hình, không khoảng trắng, không thanh cuộn)
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        overflow: hidden !important; height: 100vh !important; width: 100vw !important; margin: 0 !important; padding: 0 !important;
    }
    .main .block-container { padding: 0 !important; max-width: 100% !important; height: 100vh !important; }
    [data-testid="stHeader"], footer { display: none !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; border: none !important; z-index: 1; }
    [data-testid="stSidebar"] { z-index: 100; background-color: rgba(248, 249, 250, 0.95); border-right: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CÁC HÀM HỖ TRỢ KỸ THUẬT ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=10):
    new_rows = []
    if len(df) < 2: return df
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine_km(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        n_steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(n_steps):
            f = j / n_steps
            new_rows.append({
                'lat': p1['lat'] + (p2['lat'] - p1['lat']) * f,
                'lon': p1['lon'] + (p2['lon'] - p1['lon']) * f,
                'r6': p1.get('bán kính gió mạnh cấp 6 (km)', 0)*(1-f) + p2.get('bán kính gió mạnh cấp 6 (km)', 0)*f,
                'r10': p1.get('bán kính gió mạnh cấp 10 (km)', 0)*(1-f) + p2.get('bán kính gió mạnh cấp 10 (km)', 0)*f,
                'rc': p1.get('bán kính tâm (km)', 0)*(1-f) + p2.get('bán kính tâm (km)', 0)*f
            })
    new_rows.append(df.iloc[-1].to_dict())
    return pd.DataFrame(new_rows)

# --- 3. MODULE CODE NỀN (BASE MAP) ---
def create_base_map():
    # Tạo bản đồ nền tập trung vào Biển Đông
    m = folium.Map(location=[17.5, 115.0], zoom_start=6, tiles="OpenStreetMap", control_scale=True)
    
    # Lưới kinh vĩ độ (Giữ nguyên yêu cầu của bạn)
    for lon in range(100, 141, 5):
        folium.PolyLine([[0, lon], [40, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    for lat in range(0, 41, 5):
        folium.PolyLine([[lat, 100], [lat, 140]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    
    # Tiện ích hệ thống
    MousePosition().add_to(m)
    MeasureControl(primary_length_unit='kilometers').add_to(m)
    SimpleScreenshot().add_to(m) # Option tải ảnh PNG trực tiếp trên bản đồ
    
    return m

# --- 4. CÁC MODULE CON (Dùng để nhúng vào nền) ---

def draw_storm_wind_zones(fg, df_raw):
    """Module con: Nội suy và vẽ vùng gió"""
    df_dense = densify_track(df_raw)
    polys_r6, polys_r10, polys_rc = [], [], []
    geo = geodesic.Geodesic()
    for _, row in df_dense.iterrows():
        for r, target in [(row.get('r6', 0), polys_r6), (row.get('r10', 0), polys_r10), (row.get('rc', 0), polys_rc)]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                target.append(Polygon(circle))
    
    u6, u10, uc = unary_union(polys_r6), unary_union(polys_r10), unary_union(polys_rc)
    for geom, color, op in [(u6, COL_R6, 0.4), (u10, COL_R10, 0.5), (uc, COL_RC, 0.6)]:
        if geom and not geom.is_empty:
            folium.GeoJson(mapping(geom), style_function=lambda x,c=color,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg)

def draw_storm_path(fg, df, color="black"):
    """Module con: Vẽ đường kẻ quỹ đạo"""
    points = df[['lat', 'lon']].values.tolist()
    folium.PolyLine(points, color=color, weight=2, opacity=0.8).add_to(fg)

# --- 5. LOGIC ĐIỀU KHIỂN CHÍNH (OPTION 1 & 2) ---

m = create_base_map()
st.sidebar.title("🎛️ Bảng Điều Khiển")

if os.path.exists(DATA_FOLDER):
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    # --- OPTION 1: CHỌN BÃO HIỆN TẠI TỪ THƯ MỤC ---
    st.sidebar.subheader("📍 OPTION 1: Bão hiện tại")
    selected_current = st.sidebar.multiselect("Chọn file bão đang hoạt động:", options=all_files, default=all_files[:1] if all_files else [])
    
    # --- OPTION 2: LỌC DỮ LIỆU TỔNG HỢP ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕰️ OPTION 2: Lọc dữ liệu")
    
    full_data = []
    for f in all_files:
        try:
            df_tmp = pd.read_excel(os.path.join(DATA_FOLDER, f), sheet_name='besttrack')
            full_data.append(df_tmp)
        except: pass
    
    if full_data:
        combined_df = pd.concat(full_data).dropna(subset=['lat', 'lon'])
        combined_df[['lat', 'lon']] = combined_df[['lat', 'lon']].apply(pd.to_numeric)

        # Thanh cuộn tùy chọn lọc
        sel_nums = st.sidebar.multiselect("Lọc theo Số hiệu/Tên:", options=sorted(combined_df['Số hiệu'].unique().tolist()))
        sel_bf = st.sidebar.slider("Lọc theo Cấp gió (BF):", 0, 18, (0, 18))
        sel_pmin = st.sidebar.slider("Lọc theo Khí áp (Pmin):", 900, 1015, (900, 1015))

        # --- THỰC THI NHÚNG MODULE VÀO NỀN ---
        
        # Nhúng bão hiện tại (Vẽ màu Đỏ)
        for f_name in selected_current:
            df_storm = pd.read_excel(os.path.join(DATA_FOLDER, f_name), sheet_name='besttrack')
            df_storm_curr = df_storm[df_storm['Thời điểm'].str.contains("hiện tại|dự báo", case=False, na=False)]
            if not df_storm_curr.empty:
                fg_curr = folium.FeatureGroup(name=f"Hiện tại: {f_name}")
                draw_storm_path(fg_curr, df_storm_curr, color="red")
                draw_storm_wind_zones(fg_curr, df_storm_curr)
                fg_curr.add_to(m)

        # Nhúng bão đã lọc (Vẽ màu Xanh)
        if sel_nums:
            df_filtered = combined_df[
                (combined_df['Thời điểm'].str.contains("quá khứ", case=False, na=False)) &
                (combined_df['Số hiệu'].isin(sel_nums)) &
                (combined_df['cường độ (cấp BF)'].between(sel_bf[0], sel_bf[1])) &
                (combined_df['Pmin (mb)'].between(sel_pmin[0], sel_pmin[1]))
            ]
            if not df_filtered.empty:
                fg_past = folium.FeatureGroup(name="Dữ liệu lọc")
                draw_storm_path(fg_past, df_filtered, color="blue")
                fg_past.add_to(m)

# Hiển thị trình điều khiển Layer
folium.LayerControl(position='topleft').add_to(m)

# Render bản đồ tràn màn hình
st_folium(m, width=2500, height=1200, use_container_width=True)
