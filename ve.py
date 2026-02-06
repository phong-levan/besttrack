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

# Tích hợp các công cụ hỗ trợ (Plugins)
try:
    from folium.plugins import SimpleScreenshot, MousePosition, MeasureControl, Fullscreen
    HAS_PLUGINS = True
except ImportError:
    HAS_PLUGINS = False

# --- 1. CẤU HÌNH HỆ THỐNG ---
DATA_FOLDER = "besttrack"  
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(
    page_title="Hệ thống Dashboard Khí tượng", 
    layout="wide", 
    initial_sidebar_state="collapsed" # Thu gọn sidebar để giống iWeather
)

# --- 2. CSS INJECTION: GIAO DIỆN DASHBOARD CHUYÊN NGHIỆP ---
st.markdown("""
    <style>
    /* Làm bản đồ tràn viền 100% */
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
    
    /* Ép bản đồ dán chặt vào nền */
    iframe { 
        position: fixed; 
        top: 0; left: 0; 
        width: 100vw !important; 
        height: 100vh !important; 
        border: none !important; 
        z-index: 1 !important; 
    }

    /* Tùy chỉnh ô vuông LayerControl để nó hiện thị đẹp hơn giống iWeather */
    .leaflet-control-layers {
        box-shadow: 0 1px 5px rgba(0,0,0,0.4) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 5px !important;
        padding: 10px !important;
        font-family: 'Arial', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODULE CƠ SỞ (BASE MAP) ---
def create_base_map():
    # Khởi tạo bản đồ nền
    m = folium.Map(location=[17.5, 115.0], zoom_start=6, tiles="OpenStreetMap", control_scale=True)
    
    # Vẽ lưới và nhãn kinh vĩ độ (Nhãn này sẽ luôn hiện trên nền)
    for lon in range(100, 145, 5):
        folium.PolyLine([[0, lon], [45, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
        folium.Marker([1, lon], icon=folium.DivIcon(html=f'<div style="font-size: 8pt; color: gray;">{lon}°E</div>')).add_to(m)
    for lat in range(0, 45, 5):
        folium.PolyLine([[lat, 100], [lat, 145]], color='gray', weight=0.5, opacity=0.3).add_to(m)
        folium.Marker([lat, 101], icon=folium.DivIcon(html=f'<div style="font-size: 8pt; color: gray;">{lat}°N</div>')).add_to(m)
    
    if HAS_PLUGINS:
        MousePosition().add_to(m)
        MeasureControl(position='bottomleft').add_to(m)
        SimpleScreenshot().add_to(m)
        Fullscreen().add_to(m)
    return m

# --- 4. MODULE VẼ BÃO (STORM ENGINE) ---
def add_storm_to_map(map_obj, df, layer_name, color, show_swaths=False):
    # Tạo một FeatureGroup (đây chính là cái sẽ hiện trong bảng Layer)
    fg = folium.FeatureGroup(name=layer_name, overlay=True, control=True)
    
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

    # Vẽ quỹ đạo
    folium.PolyLine(df[['lat', 'lon']].values.tolist(), color=color, weight=2, opacity=0.8).add_to(fg)
    
    # Marker tâm bão
    last_row = df.iloc[-1]
    folium.Marker([last_row['lat'], last_row['lon']], popup=f"<b>{layer_name}</b>").add_to(fg)
    
    # Thêm toàn bộ nhóm này vào bản đồ
    fg.add_to(map_obj)

# --- 5. CHƯƠNG TRÌNH CHÍNH ---
m = create_base_map()

# Đọc dữ liệu từ thư mục và tự động đẩy vào Layer Control
if os.path.exists(DATA_FOLDER):
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    # Duyệt qua từng file bão để tạo thành các lớp riêng biệt trong ô vuông xếp lớp
    for f_name in all_files:
        try:
            path = os.path.join(DATA_FOLDER, f_name)
            df = pd.read_excel(path)
            df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce').dropna()
            
            # Phân loại màu: Nếu là file 'besttrack.xlsx' thì màu đỏ, còn lại màu xanh
            line_color = "red" if "besttrack.xlsx" in f_name else "blue"
            is_current = True if "besttrack.xlsx" in f_name else False
            
            # Đẩy vào bản đồ dưới dạng một Layer
            add_storm_to_map(m, df, f"🌀 {f_name.split('.')[0]}", line_color, show_swaths=is_current)
        except Exception as e:
            continue

# THIẾT KẾ QUAN TRỌNG: LayerControl giống iWeather
# position='topright' (Góc trên bên phải), collapsed=False (Luôn mở rộng menu)
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# Hiển thị bản đồ tràn màn hình
st_folium(m, width=2500, height=1200, use_container_width=True)
