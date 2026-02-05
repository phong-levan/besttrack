import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

# --- CẤU HÌNH MÀU SẮC (Lấy từ file của Phong) ---
COL_R6   = "#FFC0CB"  # Hồng (Bán kính gió cấp 6)
COL_R10  = "#FF6347"  # Đỏ cam (Bán kính gió cấp 10)
COL_RC   = "#90EE90"  # Xanh lá nhạt (Bán kính tâm)
COL_TRACK = "black"    # Đường đi bão

st.set_page_config(page_title="Theo dõi xoáy thuận nhiệt đới", layout="wide")
st.title("🌀 Theo dõi xoáy thuận nhiệt đới")

# Đọc dữ liệu
@st.cache_data
def load_data():
    if os.path.exists("besttrack.xlsx"):
        df = pd.read_excel("besttrack.xlsx")
        # Đảm bảo lat/lon là số thực
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        return df.dropna(subset=['lat', 'lon'])
    return None

df = load_data()

if df is not None:
    # 1. Khởi tạo bản đồ nền
    m = folium.Map(location=[15.8, 112.0], zoom_start=5, tiles="CartoDB positron")
    
    points = []
    for i, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        points.append([lat, lon])
        
        # 2. Vẽ các vùng gió (Sử dụng đúng màu Phong yêu cầu)
        # Vùng gió cấp 6
        r6 = row.get('bán kính gió mạnh cấp 6 (km)', 0)
        if r6 > 0:
            folium.Circle(
                location=[lat, lon], radius=r6*1000,
                color=COL_R6, fill=True, fill_opacity=0.3
            ).add_to(m) # Đã sửa lỗi .add_to(m)

        # Vùng gió cấp 10
        r10 = row.get('bán kính gió mạnh cấp 10 (km)', 0)
        if r10 > 0:
            folium.Circle(
                location=[lat, lon], radius=r10*1000,
                color=COL_R10, fill=True, fill_opacity=0.4
            ).add_to(m)

        # Vùng tâm bão
        rc = row.get('bán kính tâm (km)', 0)
        if rc > 0:
            folium.Circle(
                location=[lat, lon], radius=rc*1000,
                color=COL_RC, fill=True, fill_opacity=0.6
            ).add_to(m)

        # 3. Điểm tâm bão và Popup thông tin
        is_past = "quá khứ" in str(row.get('Thời điểm', '')).lower()
        marker_color = "black" if is_past else "red"
        
        folium.CircleMarker(
            location=[lat, lon], radius=4,
            color=marker_color, fill=True,
            popup=f"Cấp: {row.get('cường độ (cấp BF)', 'N/A')}<br>Pmin: {row.get('Pmin (mb)', 'N/A')} mb"
        ).add_to(m)

    # 4. Vẽ đường quỹ đạo
    if len(points) > 1:
        folium.PolyLine(points, color=COL_TRACK, weight=2, opacity=0.7).add_to(m)

    # Hiển thị lên giao diện Web
    st_folium(m, width="100%", height=600)
    st.dataframe(df)
else:
    st.error("Không tìm thấy dữ liệu 'besttrack.xlsx' trong thư mục dự án.")
