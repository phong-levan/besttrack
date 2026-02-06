# -*- coding: utf-8 -*-
import streamlit as st
import folium
from streamlit_folium import st_folium

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Hệ thống Bản đồ Nền Full Viền", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. CSS INJECTION: TRIỆT TIÊU KHOẢNG TRẮNG TUYỆT ĐỐI ---
st.markdown("""
    <style>
    /* Loại bỏ lề và thanh cuộn của trình duyệt */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        overflow: hidden !important;
        height: 100vh !important;
        width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Xóa khoảng cách mặc định của Streamlit container */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        height: 100vh !important;
    }

    /* Ẩn hoàn toàn Header, Footer và các thành phần thừa */
    [data-testid="stHeader"], footer, [data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Ép bản đồ (Iframe) dán chặt vào 4 góc màn hình */
    iframe {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HÀM KHỞI TẠO BẢN ĐỒ NỀN ---
def create_base_map():
    # Khởi tạo bản đồ tại khu vực Biển Đông
    m = folium.Map(
        location=[17.5, 114.0], 
        zoom_start=6, 
        tiles="OpenStreetMap",
        control_scale=True
    )

    # --- BỔ SUNG LƯỚI KINH VĨ ĐỘ ---
    # Vẽ các đường kinh tuyến mỗi 5 độ
    for lon in range(100, 141, 5):
        folium.PolyLine([[0, lon], [45, lon]], color='gray', weight=0.6, opacity=0.4).add_to(m)

    # Vẽ các đường vĩ tuyến mỗi 5 độ
    for lat in range(0, 41, 5):
        folium.PolyLine([[lat, 100], [lat, 145]], color='gray', weight=0.6, opacity=0.4).add_to(m)
        
    return m

# --- 4. HIỂN THỊ ---
base_map = create_base_map()

# Với CSS ở trên, thông số width/height ở đây sẽ bị CSS ghi đè để full màn hình
st_folium(
    base_map, 
    width=2500, 
    height=1200, 
    use_container_width=True
)
