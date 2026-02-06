# -*- coding: utf-8 -*-
import streamlit as st
import folium
from streamlit_folium import st_folium

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Bản đồ nền Kinh vĩ độ", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. CSS INJECTION: CHIẾN THUẬT TRÀN VIỀN TUYỆT ĐỐI ---
st.markdown("""
    <style>
    /* 1. Xóa bỏ hoàn toàn thanh cuộn và lề của trình duyệt */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        overflow: hidden !important;
        height: 100vh !important;
        width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 2. Xóa khoảng cách (padding) của container chính Streamlit */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        height: 100vh !important;
    }

    /* 3. Ẩn Header (thanh trắng trên cùng) và Footer */
    [data-testid="stHeader"], footer {
        display: none !important;
    }
    
    /* 4. Ép bản đồ Folium chiếm trọn 100% màn hình */
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

# --- 3. HÀM KHỞI TẠO BẢN ĐỒ NỀN (BASE MAP) ---
def create_base_map():
    # Khởi tạo bản đồ tại khu vực Biển Đông
    m = folium.Map(
        location=[17.5, 114.0], 
        zoom_start=6, 
        tiles="OpenStreetMap",
        control_scale=True
    )

    # --- BỔ SUNG LƯỚI KINH VĨ ĐỘ ---
    # Vẽ các đường kinh tuyến (mỗi 5 độ)
    for lon in range(100, 141, 5):
        folium.PolyLine(
            [[0, lon], [40, lon]], 
            color='gray', 
            weight=0.5, 
            opacity=0.5
        ).add_to(m)

    # Vẽ các đường vĩ tuyến (mỗi 5 độ)
    for lat in range(0, 41, 5):
        folium.PolyLine(
            [[lat, 100], [lat, 140]], 
            color='gray', 
            weight=0.5, 
            opacity=0.5
        ).add_to(m)
        
    return m

# --- 4. HIỂN THỊ ---
m = create_base_map()

# Hiển thị bản đồ tràn màn hình
st_folium(m, width=None, height=None, use_container_width=True)
