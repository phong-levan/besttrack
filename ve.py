# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

# Tích hợp các công cụ hỗ trợ
try:
    from folium.plugins import SimpleScreenshot, MousePosition, MeasureControl
    HAS_PLUGINS = True
except ImportError:
    HAS_PLUGINS = False

# --- 1. CẤU HÌNH HỆ THỐNG ---
DATA_FOLDER = "besttrack"  
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(
    page_title="Hệ thống Theo dõi Bão", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS INJECTION: FIX LỖI SIDEBAR & TRÀN VIỀN ---
st.markdown("""
    <style>
    /* Cho phép cuộn Sidebar */
    [data-testid="stSidebarUserContent"] {
        overflow-y: auto !important;
        max-height: 100vh;
    }
    
    /* Đảm bảo Sidebar luôn nằm TRÊN bản đồ */
    [data-testid="stSidebar"] {
        z-index: 999999 !important;
        background-color: white !important;
    }

    /* Bản đồ nằm dưới cùng */
    iframe {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        z-index: 1 !important;
    }

    /* Ẩn các thành phần thừa */
    [data-testid="stHeader"], footer, .main .block-container {
        display: none !important;
    }
    
    /* Hiện lại nội dung Sidebar */
    section[data-testid="stSidebar"] > div {
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODULE CƠ SỞ (BASE MAP) ---
def create_base_map():
    # Khởi tạo bản đồ khu vực Biển Đông
    m = folium.Map(location=[17.5, 115.0], zoom_start=6, tiles="OpenStreetMap")
    
    # Vẽ lưới và số kinh vĩ độ
    for lon in range(100, 145, 5):
        folium.PolyLine([[0, lon], [45, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
        folium.Marker(
            [1, lon], 
            icon=folium.DivIcon(html=f'<div style="font-size: 9pt; color: gray;">{lon}°E</div>')
        ).add_to(m)

    for lat in range(0, 45, 5):
        folium.PolyLine([[lat, 100], [lat, 145]], color='gray', weight=0.5, opacity=0.3).add_to(m)
        folium.Marker(
            [lat, 101], 
            icon=folium.DivIcon(html=f'<div style="font-size: 9pt; color: gray;">{lat}°N</div>')
        ).add_to(m)
    
    if HAS_PLUGINS:
        MousePosition().add_to(m)
        MeasureControl(primary_length_unit='kilometers').add_to(m)
        SimpleScreenshot().add_to(m) # Nút chụp ảnh PNG ở góc trái
    return m

# --- 4. MODULE VẼ BÃO (STORM ENGINE) ---
def draw_storm_layers(fg, df, color="black", show_swaths=False):
    # Vẽ bán kính gió (Swaths)
    if show_swaths and len(df) >= 2:
        polys_r6, polys_r10, polys_rc = [], [], []
        geo = geodesic.Geodesic()
        for _, row in df.iterrows():
            for r, target in [(row.get('bán kính gió mạnh cấp 6 (km)', 0), polys_r6), 
                              (row.get('bán kính gió mạnh cấp 10 (km)', 0), polys_r10), 
                              (row.get('bán kính tâm (km)', 0), polys_rc)]:
                if r > 0:
                    circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                    target.append(Polygon(circle))
        
        u6 = unary_union(polys_r6) if polys_r6 else None
        u10 = unary_union(polys_r10) if polys_r10 else None
        uc = unary_union(polys_rc) if polys_rc else None
        
        for geom, col, op in [(u6, COL_R6, 0.4), (u10, COL_R10, 0.5), (uc, COL_RC, 0.6)]:
            if geom and not geom.is_empty:
                folium.GeoJson(mapping(geom), style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg)

    # Vẽ đường đi
    points = df[['lat', 'lon']].values.tolist()
    folium.PolyLine(points, color=color, weight=2, opacity=0.8).add_to(fg)
    
    # Vẽ Marker tâm bão
    last_row = df.iloc[-1]
    folium.Marker(
        [last_row['lat'], last_row['lon']], 
        popup=f"Bão: {last_row.get('Số hiệu')}"
    ).add_to(fg)

# --- 5. CHƯƠNG TRÌNH CHÍNH (SIDEBAR CONTROLS) ---
m = create_base_map()
st.sidebar.title("🌪️ Quản lý Dữ liệu Bão")

# Kiểm tra thư mục dữ liệu
if not os.path.exists(DATA_FOLDER):
    st.sidebar.error(f"❌ Không tìm thấy thư mục '{DATA_FOLDER}'")
else:
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    # --- OPTION 1: DÀNH CHO BESTTRACK.XLSX ---
    st.sidebar.header("📍 Option 1: Bão hiện tại")
    f_opt1 = st.sidebar.selectbox("Chọn file (mặc định besttrack.xlsx):", options=all_files, index=0 if "besttrack.xlsx" in all_files else 0)
    
    if f_opt1:
        df1 = pd.read_excel(os.path.join(DATA_FOLDER, f_opt1))
        df1[['lat', 'lon']] = df1[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
        df1 = df1.dropna(subset=['lat', 'lon'])
        
        fg1 = folium.FeatureGroup(name="Option 1: Hiện trạng")
        draw_storm_layers(fg1, df1, color="red", show_swaths=True)
        fg1.add_to(m)

    # --- OPTION 2: DÀNH CHO BESTTRACK_CAPGIO.XLSX + BỘ LỌC ---
    st.sidebar.markdown("---")
    st.sidebar.header("🕰️ Option 2: Lọc bão quá khứ")
    f_opt2 = st.sidebar.selectbox("Chọn file dữ liệu:", options=all_files, index=0 if "besttrack_capgio.xlsx" in all_files else 0)
    
    if f_opt2:
        df2 = pd.read_excel(os.path.join(DATA_FOLDER, f_opt2))
        df2[['lat', 'lon']] = df2[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
        df2['Ngày - giờ'] = pd.to_datetime(df2['Ngày - giờ'], errors='coerce')
        df2 = df2.dropna(subset=['lat', 'lon'])
        
        # Tạo bộ lọc theo thời gian và tên
        years = sorted(df2['Ngày - giờ'].dt.year.dropna().unique().astype(int))
        sel_year = st.sidebar.multiselect("Năm:", options=years)
        
        names = sorted(df2['Số hiệu'].unique())
        sel_names = st.sidebar.multiselect("Tên/Số hiệu bão:", options=names)
        
        sel_bf = st.sidebar.slider("Cấp gió (BF):", 0, 18, (0, 18))
        sel_pmin = st.sidebar.slider("Khí áp (Pmin):", 900, 1015, (900, 1015))

        # Áp dụng lọc
        mask = (df2['cường độ (cấp BF)'].between(sel_bf[0], sel_bf[1])) & \
               (df2['Pmin (mb)'].between(sel_pmin[0], sel_pmin[1]))
        
        if sel_year: mask &= df2['Ngày - giờ'].dt.year.isin(sel_year)
        if sel_names: mask &= df2['Số hiệu'].isin(sel_names)
        
        df2_filtered = df2[mask]
        
        if not df2_filtered.empty:
            fg2 = folium.FeatureGroup(name="Option 2: Dữ liệu lọc")
            draw_storm_layers(fg2, df2_filtered, color="blue", show_swaths=False)
            fg2.add_to(m)

folium.LayerControl(position='topleft').add_to(m)

# Hiển thị bản đồ (CSS sẽ tự động ép full màn hình)
st_folium(m, width=2500, height=1200, use_container_width=True)
