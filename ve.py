import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
from math import radians, sin, cos, asin, sqrt

# --- CẤU HÌNH MÀU SẮC & ĐƯỜNG DẪN ICON ---
ICON_DIR = "icon"
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"
COL_TRACK = "black"

st.set_page_config(page_title="Hệ thống Theo dõi Bão - Phong Le", layout="wide")
st.title("🌀 Bản đồ Bão Tương tác với Biểu tượng Tùy chỉnh")

# --- HÀM LẤY ĐƯỜNG DẪN ICON DỰA TRÊN TRẠNG THÁI ---
def get_storm_icon_path(row):
    status = "daqua" if "quá khứ" in str(row.get('Thời điểm', '')).lower() else "dubao"
    wind_speed = row.get('cường độ (cấp BF)', 0)
    
    # Logic phân loại icon giống như file ve.py của bạn
    if pd.isna(wind_speed) or wind_speed < 6:
        name = f"vungthap{status}.png"
    elif wind_speed < 8:
        name = "atnddaqua.PNG" if status == "daqua" else "atnd.PNG"
    elif wind_speed <= 11:
        name = "bnddaqua.PNG" if status == "daqua" else "bnd.PNG"
    else:
        name = "sieubaodaqua.PNG" if status == "daqua" else "sieubao.PNG"
    
    return os.path.join(ICON_DIR, name)

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists("besttrack.xlsx"):
    df_raw = pd.read_excel("besttrack.xlsx")
    df_raw[['lat', 'lon']] = df_raw[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    df_raw = df_raw.dropna(subset=['lat', 'lon'])

    m = folium.Map(location=[15.8, 112.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Vẽ hành lang gió (Nội suy 10km như trước)
    # ... (giữ nguyên phần densify và vẽ Circle để tạo dải trong suốt) ...

    # 2. Vẽ Icon tâm bão tại các điểm gốc
    for _, row in df_raw.iterrows():
        icon_path = get_storm_icon_path(row)
        
        if os.path.exists(icon_path):
            # Tạo icon tùy chỉnh từ file trong thư mục icon/
            icon = folium.CustomIcon(
                icon_path,
                icon_size=(30, 30) if "vungthap" not in icon_path else (15, 15)
            )
            
            folium.Marker(
                location=[row['lat'], row['lon']],
                icon=icon,
                popup=folium.Popup(f"Thời gian: {row.get('Ngày - giờ', 'N/A')}<br>Cấp: {row.get('cường độ (cấp BF)', 'N/A')}", max_width=200)
            ).add_to(m)
        else:
            # Fallback nếu không tìm thấy file icon
            folium.CircleMarker(
                location=[row['lat'], row['lon']], radius=4, color="black", fill=True
            ).add_to(m)

    st_folium(m, width="100%", height=600)
else:
    st.error("Không tìm thấy file besttrack.xlsx")
