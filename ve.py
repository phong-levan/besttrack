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

# Cố gắng tải các công cụ mở rộng (Plugins)
try:
    from folium.plugins import SimpleScreenshot, MousePosition, MeasureControl
    HAS_PLUGINS = True
except ImportError:
    HAS_PLUGINS = False

# --- 1. CẤU HÌNH HỆ THỐNG ---
DATA_FOLDER = "besttrack"  
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

# Đảm bảo initial_sidebar_state="expanded" để thanh tùy chọn luôn hiện ra
st.set_page_config(page_title="Hệ thống Theo dõi Bão", layout="wide", initial_sidebar_state="expanded")

# CSS: GIỮ NỀN TRÀN VIỀN VÀ CHO PHÉP CUỘN THANH TÙY CHỌN
st.markdown("""
    <style>
    [data-testid="stSidebarUserContent"] {
        overflow-y: auto !important;
        max-height: 100vh;
    }
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important; 
        height: 100vh !important; 
        width: 100vw !important; 
        margin: 0 !important; 
        padding: 0 !important;
    }
    .main .block-container { 
        padding: 0 !important; 
        max-width: 100% !important; 
        height: 100vh !important; 
    }
    [data-testid="stHeader"], footer { display: none !important; }
    iframe { 
        position: fixed; 
        top: 0; left: 0; 
        width: 100vw !important; 
        height: 100vh !important; 
        border: none !important; 
        z-index: 1; 
    }
    [data-testid="stSidebar"] { z-index: 100; background-color: rgba(248, 249, 250, 0.95); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODULE CODE NỀN (BASE MAP) ---
def create_base_map():
    # Khởi tạo bản đồ
    m = folium.Map(location=[17.5, 115.0], zoom_start=5, tiles="OpenStreetMap")
    
    # Vẽ lưới kinh vĩ độ và đánh số nhãn
    # Kinh độ (Longitude)
    for lon in range(-180, 181, 10):
        folium.PolyLine([[-90, lon], [90, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
        # Ghi số kinh độ ở gần xích đạo (vĩ độ 0)
        folium.map.Marker(
            [0, lon],
            icon=folium.DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 8pt; color: gray;">{lon}E</div>',
            )
        ).add_to(m)

    # Vĩ độ (Latitude)
    for lat in range(-90, 91, 10):
        folium.PolyLine([[lat, -180], [lat, 180]], color='gray', weight=0.5, opacity=0.3).add_to(m)
        # Ghi số vĩ độ ở gần kinh độ 100
        folium.map.Marker(
            [lat, 100],
            icon=folium.DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 8pt; color: gray;">{lat}N</div>',
            )
        ).add_to(m)
    
    if HAS_PLUGINS:
        MousePosition().add_to(m)
        MeasureControl(primary_length_unit='kilometers').add_to(m)
        SimpleScreenshot().add_to(m)
    
    return m

# --- 3. MODULE VẼ BÃO (STORM ENGINE) ---
def draw_storm_module(fg, df_raw, color="black", show_swaths=False):
    if show_swaths and len(df_raw) >= 2:
        polys_r6, polys_r10, polys_rc = [], [], []
        geo = geodesic.Geodesic()
        for _, row in df_raw.iterrows():
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

    points = df_raw[['lat', 'lon']].values.tolist()
    folium.PolyLine(points, color=color, weight=2, opacity=0.8).add_to(fg)

# --- 4. LOGIC ĐIỀU KHIỂN (MAIN) ---
m = create_base_map()

# Tạo Sidebar
st.sidebar.title("🎛️ Bảng Điều Khiển")

if os.path.exists(DATA_FOLDER):
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    st.sidebar.subheader("📍 Option 1: Bão hiện tại")
    sel_curr = st.sidebar.multiselect("Chọn file bão đang hoạt động:", options=all_files, default=all_files[:1] if all_files else [])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕰️ Option 2: Lọc bão quá khứ")
    
    all_dfs = []
    for f in all_files:
        try:
            tmp = pd.read_excel(os.path.join(DATA_FOLDER, f), sheet_name='besttrack')
            all_dfs.append(tmp)
        except: pass
    
    if all_dfs:
        combined = pd.concat(all_dfs).dropna(subset=['lat', 'lon'])
        combined[['lat', 'lon']] = combined[['lat', 'lon']].apply(pd.to_numeric)
        
        sel_nums = st.sidebar.multiselect("Chọn Số hiệu bão:", options=sorted(combined['Số hiệu'].unique().tolist()))
        bf_range = st.sidebar.slider("Cấp gió (BF):", 0, 18, (0, 18))

        for f_name in sel_curr:
            df_storm = pd.read_excel(os.path.join(DATA_FOLDER, f_name), sheet_name='besttrack')
            df_curr = df_storm[df_storm['Thời điểm'].str.contains("hiện tại|dự báo", case=False, na=False)]
            if not df_curr.empty:
                fg = folium.FeatureGroup(name=f"Hiện tại: {f_name}")
                draw_storm_module(fg, df_curr, color="red", show_swaths=True)
                fg.add_to(m)

        if sel_nums:
            df_past = combined[(combined['Số hiệu'].isin(sel_nums)) & (combined['cường độ (cấp BF)'].between(bf_range[0], bf_range[1]))]
            if not df_past.empty:
                fg_p = folium.FeatureGroup(name="Dữ liệu lọc")
                draw_storm_module(fg_p, df_past, color="blue", show_swaths=False)
                fg_p.add_to(m)
else:
    st.sidebar.error(f"Thư mục '{DATA_FOLDER}' không tồn tại.")

folium.LayerControl(position='topleft').add_to(m)
st_folium(m, width=2500, height=1200, use_container_width=True)
