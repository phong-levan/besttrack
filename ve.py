# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import base64
from math import radians, sin, cos, asin, sqrt

# --- 1. CẤU HÌNH HỆ THỐNG & CSS ---
st.set_page_config(page_title="Hệ thống Giám sát Bão đa tầng", layout="wide")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { overflow: hidden !important; height: 100vh; width: 100vw; margin: 0; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; height: 100vh !important; }
    [data-testid="stHeader"], footer { display: none !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; border: none !important; }
    [data-testid="stSidebar"] { z-index: 100; background-color: rgba(248, 249, 250, 0.95); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM BẢN ĐỒ CHUNG (BASE MAP) ---
def create_base_map():
    m = folium.Map(location=[17.5, 115.0], zoom_start=6, tiles="OpenStreetMap")
    # Lưới kinh vĩ độ
    for lon in range(100, 141, 5):
        folium.PolyLine([[0, lon], [40, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    for lat in range(0, 41, 5):
        folium.PolyLine([[lat, 100], [lat, 140]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    return m

# --- 3. HÀM CON VẼ DỮ LIỆU BÃO (LAYER MODULE) ---
def add_storm_layer(map_obj, df, layer_name, color):
    fg = folium.FeatureGroup(name=layer_name)
    points = df[['lat', 'lon']].values.tolist()
    # Vẽ quỹ đạo
    folium.PolyLine(points, color=color, weight=3, opacity=0.7).add_to(fg)
    # Vẽ điểm marker
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            color=color,
            fill=True,
            popup=f"Bão: {row.get('Số hiệu', 'N/A')}<br>Cấp: {row.get('cường độ (cấp BF)', 0)}<br>Pmin: {row.get('Pmin (mb)', 0)}"
        ).add_to(fg)
    fg.add_to(map_obj)

# --- 4. CHƯƠNG TRÌNH CHÍNH ---

# Đọc dữ liệu
DATA_FILE = "besttrack_capgio.xlsx - besttrack.csv" # Đường dẫn file bạn đã cung cấp
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])

    # Giao diện Sidebar Lọc dữ liệu
    st.sidebar.title("🌪️ Tùy chọn bản đồ")
    
    # --- THANH CUỘN 1: BÃO HIỆN TẠI/DỰ BÁO ---
    st.sidebar.subheader("📍 Trạng thái bão hiện tại")
    show_current = st.sidebar.toggle("Hiển thị bão hiện tại/dự báo", value=True)
    
    # --- THANH CUỘN 2: LỌC DỮ LIỆU QUÁ KHỨ ---
    st.sidebar.subheader("🕰️ Lọc dữ liệu bão")
    
    # Lọc theo Tên/Số hiệu
    storm_list = df['Số hiệu'].unique().tolist()
    selected_storms = st.sidebar.multiselect("Lọc theo Số hiệu bão:", options=storm_list, default=storm_list[:1])
    
    # Lọc theo Cấp gió (Slider)
    max_bf = int(df['cường độ (cấp BF)'].max())
    bf_range = st.sidebar.slider("Lọc theo cấp gió (BF):", 0, max_bf, (0, max_bf))
    
    # Lọc theo Khí áp (Slider)
    pmin_min = int(df['Pmin (mb)'].min())
    pmin_max = int(df['Pmin (mb)'].max())
    pmin_range = st.sidebar.slider("Lọc theo khí áp (Pmin):", pmin_min, pmin_max, (pmin_min, pmin_max))

    # Xử lý Logic lọc dữ liệu
    df_filtered = df[
        (df['Số hiệu'].isin(selected_storms)) &
        (df['cường độ (cấp BF)'].between(bf_range[0], bf_range[1])) &
        (df['Pmin (mb)'].between(pmin_range[0], pmin_range[1]))
    ]

    # Khởi tạo bản đồ
    m = create_base_map()

    # Vẽ Layer bão hiện tại (Nếu chọn)
    if show_current:
        df_current = df[df['Thời điểm'].str.contains("hiện tại|dự báo", case=False, na=False)]
        if not df_current.empty:
            add_storm_layer(m, df_current, "Bão hiện tại/Dự báo", "red")

    # Vẽ Layer bão quá khứ theo bộ lọc
    df_past = df_filtered[df_filtered['Thời điểm'].str.contains("quá khứ", case=False, na=False)]
    if not df_past.empty:
        add_storm_layer(m, df_past, "Dữ liệu bão lọc tùy chọn", "blue")

    # Layer Control trực tiếp trên bản đồ
    folium.LayerControl(position='topleft').add_to(m)

    # Hiển thị
    st_folium(m, width=2000, height=1200, use_container_width=True)

else:
    st.error("Không tìm thấy file dữ liệu. Hãy kiểm tra tên file csv.")
