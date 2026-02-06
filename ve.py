# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

# --- 1. CẤU HÌNH ---
DATA_FOLDER = "besttrack"
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(page_title="Storm Dashboard", layout="wide", initial_sidebar_state="collapsed")

# CSS: Ép bản đồ tràn màn hình 100%
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important; 
        height: 100vh !important; 
        margin: 0 !important; padding: 0 !important;
    }
    .main .block-container { padding: 0 !important; max-width: 100% !important; height: 100vh !important; }
    [data-testid="stHeader"], footer { display: none !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; border: none !important; z-index: 1 !important; }
    /* Style cho bảng LayerControl giống iWeather */
    .leaflet-control-layers-list { font-size: 14px; font-weight: bold; line-height: 2; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM VẼ VÙNG GIÓ (SWATHS) ---
def get_storm_swaths(df):
    polys_r6, polys_r10, polys_rc = [], [], []
    geo = geodesic.Geodesic()
    for _, row in df.iterrows():
        for r, target in [(row.get('bán kính gió mạnh cấp 6 (km)', 0), polys_r6), 
                          (row.get('bán kính gió mạnh cấp 10 (km)', 0), polys_r10), 
                          (row.get('bán kính tâm (km)', 0), polys_rc)]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                target.append(Polygon(circle))
    return unary_union(polys_r6), unary_union(polys_r10), unary_union(polys_rc)

# --- 3. KHỞI TẠO BẢN ĐỒ ---
# Tạo bản đồ nền
m = folium.Map(location=[17.5, 115.0], zoom_start=6, tiles="OpenStreetMap")

# Thêm số kinh vĩ độ lên lưới
for lon in range(100, 145, 5):
    folium.PolyLine([[0, lon], [45, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    folium.map.Marker([1, lon], icon=folium.DivIcon(html=f'<div style="font-size: 8pt; color: gray;">{lon}E</div>')).add_to(m)

for lat in range(0, 45, 5):
    folium.PolyLine([[lat, 100], [lat, 145]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    folium.map.Marker([lat, 100.5], icon=folium.DivIcon(html=f'<div style="font-size: 8pt; color: gray;">{lat}N</div>')).add_to(m)

# --- 4. ĐỌC DỮ LIỆU VÀ TẠO LAYER TÙY CHỌN ---
if os.path.exists(DATA_FOLDER):
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    for f in files:
        try:
            df = pd.read_excel(os.path.join(DATA_FOLDER, f))
            # Chuẩn hóa cột lat/lon
            df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=['lat', 'lon'])
            
            # Tên lớp hiển thị trong ô vuông
            layer_name = f"🌀 {f.replace('.xlsx', '')}"
            
            # Tạo FeatureGroup cho từng cơn bão
            fg = folium.FeatureGroup(name=layer_name, show=True if "besttrack" in f else False)
            
            # Vẽ Swaths (Nếu là file hiện tại)
            if "besttrack" in f and not "capgio" in f:
                u6, u10, uc = get_storm_swaths(df)
                for geom, col, op in [(u6, COL_R6, 0.4), (u10, COL_R10, 0.5), (uc, COL_RC, 0.6)]:
                    if geom and not geom.is_empty:
                        folium.GeoJson(mapping(geom), style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg)
            
            # Vẽ đường quỹ đạo
            folium.PolyLine(df[['lat', 'lon']].values.tolist(), color="red" if "besttrack" in f else "blue", weight=2).add_to(fg)
            
            # Thêm vào bản đồ chính
            fg.add_to(m)
        except:
            continue

# --- 5. HIỂN THỊ Ô TÙY CHỌN (LAYER CONTROL) ---
# Đây là "Ô vuông xếp lớp" bạn yêu cầu - nó sẽ chứa danh sách các bão
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# Hiển thị
st_folium(m, width=2500, height=1200, use_container_width=True)
