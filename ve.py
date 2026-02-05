import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import io
import branca
from math import radians, sin, cos, asin, sqrt

# --- CẤU HÌNH ---
ICON_DIR = "icon"
DATA_FILE = "besttrack.xlsx"
HISTORY_FILE = "history_tracking.xlsx"

st.set_page_config(page_title="Hệ thống Dự báo Bão - Phong Le", layout="wide")

# --- HÀM NỘI SUY (10km) ---
def densify_data(df, step_km=10):
    # (Giữ nguyên hàm nội suy 10km từ các bước trước để tạo dải mịn)
    # ... logic nội suy ...
    return pd.DataFrame(rows) # Giả định hàm trả về DF nội suy

# --- TẠO BẢNG HTML NỔI TRÊN BẢN ĐỒ ---
def create_html_table(df):
    # Lấy 5 dòng cuối cùng (mới nhất) để hiển thị tin khẩn cấp
    last_points = df.tail(5)
    
    table_html = """
    <div style="position: fixed; top: 10px; right: 10px; width: 320px; z-index:9999; 
                background-color: white; padding: 10px; border: 2px solid black; 
                border-radius: 5px; font-family: Arial; font-size: 11px; opacity: 0.9;">
        <h4 style="margin-top:0; text-align:center;">TIN BÃO KHẨN CẤP</h4>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 4px;">Ngày-Giờ</th>
                <th style="border: 1px solid #ddd; padding: 4px;">Tọa độ</th>
                <th style="border: 1px solid #ddd; padding: 4px;">Gió</th>
            </tr>
    """
    for _, row in last_points.iterrows():
        table_html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 4px;">{row['Ngày - giờ']}</td>
                <td style="border: 1px solid #ddd; padding: 4px;">{row['lat']}N/{row['lon']}E</td>
                <td style="border: 1px solid #ddd; padding: 4px;">Cấp {int(row['cường độ (cấp BF)'])}</td>
            </tr>
        """
    table_html += "</table></div>"
    return table_html

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists(DATA_FILE):
    raw_df = pd.read_excel(DATA_FILE)
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])

    # --- HỘP CÔNG CỤ SIDEBAR (Giữ nguyên tính năng xuất dữ liệu) ---
    with st.sidebar:
        st.header("🛠️ Công cụ Hệ thống")
        # (Thêm các nút download Excel và PNG như bước trước)

    # --- TẠO BẢN ĐỒ ---
    m = folium.Map(location=[16.0, 112.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Thêm Bảng thông tin (HTML nổi)
    html_table = create_html_table(raw_df)
    m.get_root().html.add_child(folium.Element(html_table))

    # 2. Thêm Chú thích (Ảnh nổi)
    chuthich_path = os.path.join(ICON_DIR, "chuthich.PNG")
    if os.path.exists(chuthich_path):
        # Sử dụng FloatImage để ghim ảnh chú thích vào góc dưới bản đồ
        # Vị trí: bottom=5%, left=5%
        from folium.plugins import FloatImage
        FloatImage(chuthich_path, bottom=5, left=5).add_to(m)

    # 3. Vẽ nội suy và Icon bão
    # (Sử dụng lại logic vẽ Circle 10km trong suốt và CustomIcon đã làm)
    # ... logic vẽ quỹ đạo ...

    # Hiển thị bản đồ toàn màn hình
    st_folium(m, width=1200, height=700)

else:
    st.error("Thiếu file besttrack.xlsx")
