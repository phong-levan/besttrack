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
    [data-testid="stSidebarUserContent"] { overflow-y: auto !important; max-height: 100vh; }
    [data-testid="stSidebar"] { z-index: 999999 !important; background-color: white !important; }
    iframe {
        position: fixed; top: 0; left: 0;
        width: 100vw !important; height: 100vh !important;
        border: none !important; z-index: 1 !important;
    }
    [data-testid="stHeader"], footer, .main .block-container { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODULE CƠ SỞ (BASE MAP) ---
def create_base_map():
    m = folium.Map(location=[17.5, 115.0], zoom_start=6, tiles="OpenStreetMap")
    
    # Vẽ lưới và nhãn kinh vĩ độ
    for lon in range(100, 145, 5):
        folium.PolyLine([[0, lon], [45, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
        folium.Marker([1, lon], icon=folium.DivIcon(html=f'<div style="font-size: 9pt; color: gray;">{lon}°E</div>')).add_to(m)
    for lat in range(0, 45, 5):
        folium.PolyLine([[lat, 100], [lat, 145]], color='gray', weight=0.5, opacity=0.3).add_to(m)
        folium.Marker([lat, 101], icon=folium.DivIcon(html=f'<div style="font-size: 9pt; color: gray;">{lat}°N</div>')).add_to(m)
    
    if HAS_PLUGINS:
        MousePosition().add_to(m)
        MeasureControl(primary_length_unit='kilometers').add_to(m)
        SimpleScreenshot().add_to(m)
    return m

# --- 4. MODULE VẼ BÃO (STORM ENGINE) ---
def draw_storm_to_fg(fg, df, color="black", show_swaths=False):
    """Vẽ dữ liệu vào một FeatureGroup để hiện trong ô vuông xếp lớp"""
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
        
        u6, u10, uc = unary_union(polys_r6), unary_union(polys_r10), unary_union(polys_rc)
        for geom, col, op in [(u6, COL_R6, 0.4), (u10, COL_R10, 0.5), (uc, COL_RC, 0.6)]:
            if geom and not geom.is_empty:
                folium.GeoJson(mapping(geom), style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg)

    folium.PolyLine(df[['lat', 'lon']].values.tolist(), color=color, weight=2, opacity=0.8).add_to(fg)
    last_row = df.iloc[-1]
    folium.Marker([last_row['lat'], last_row['lon']], popup=f"Bão: {last_row.get('Số hiệu')}").add_to(fg)

# --- 5. CHƯƠNG TRÌNH CHÍNH ---
m = create_base_map()
st.sidebar.title("🌪️ Bộ lọc Dữ liệu")

if os.path.exists(DATA_FOLDER):
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    # --- OPTION 1: BÃO HIỆN TẠI ---
    st.sidebar.header("📍 Option 1: Bão hiện tại")
    sel_files_opt1 = st.sidebar.multiselect("Chọn file hiển thị trong lớp:", options=all_files, default=all_files[:1] if all_files else [])
    
    for f_name in sel_files_opt1:
        df1 = pd.read_excel(os.path.join(DATA_FOLDER, f_name))
        df1[['lat', 'lon']] = df1[['lat', 'lon']].apply(pd.to_numeric, errors='coerce').dropna()
        # Tạo FeatureGroup - Tên này sẽ hiện trong ô vuông xếp lớp
        fg_curr = folium.FeatureGroup(name=f"🔴 Hiện tại: {f_name}")
        draw_storm_to_fg(fg_curr, df1, color="red", show_swaths=True)
        fg_curr.add_to(m)

    # --- OPTION 2: LỌC QUÁ KHỨ ---
    st.sidebar.markdown("---")
    st.sidebar.header("🕰️ Option 2: Lọc quá khứ")
    f_opt2 = st.sidebar.selectbox("File dữ liệu quá khứ:", options=all_files, index=0 if len(all_files)>1 else 0)
    
    if f_opt2:
        df2 = pd.read_excel(os.path.join(DATA_FOLDER, f_opt2))
        df2[['lat', 'lon']] = df2[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
        df2['Ngày - giờ'] = pd.to_datetime(df2['Ngày - giờ'], errors='coerce')
        
        sel_names = st.sidebar.multiselect("Lọc bão:", options=sorted(df2['Số hiệu'].unique()))
        sel_bf = st.sidebar.slider("Cấp gió (BF):", 0, 18, (0, 18))

        if sel_names:
            df2_filt = df2[(df2['Số hiệu'].isin(sel_names)) & (df2['cường độ (cấp BF)'].between(sel_bf[0], sel_bf[1]))]
            if not df2_filt.empty:
                # Tạo FeatureGroup cho dữ liệu đã lọc
                fg_past = folium.FeatureGroup(name=f"🔵 Quá khứ: {f_opt2} (Đã lọc)")
                draw_storm_to_fg(fg_past, df2_filt, color="blue", show_swaths=False)
                fg_past.add_to(m)

# LỆNH QUAN TRỌNG: Hiển thị ô vuông xếp lớp (LayerControl)
folium.LayerControl(position='topright', collapsed=False).add_to(m)

st_folium(m, width=2500, height=1200, use_container_width=True)
