import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="Theo dõi xoáy thuận nhiệt đới", layout="wide")
st.title("🌀 Bản đồ theo dõi xoáy thuận nhiệt đới")

# Đọc dữ liệu
@st.cache_data
def load_data():
    if os.path.exists("besttrack.xlsx"):
        return pd.read_excel("besttrack.xlsx")
    return None

df = load_data()

if df is not None:
    # Tạo bản đồ thu phóng
    m = folium.Map(location=[15.8, 112.0], zoom_start=5, tiles="CartoDB positron")
    
    points = []
    for i, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        if pd.notna(lat) and pd.notna(lon):
            points.append([lat, lon])
            color = "black" if "quá khứ" in str(row.get('Thời điểm', '')).lower() else "red"
            
            # Thêm Marker cho từng điểm bão
            folium.CircleMarker(
                location=[lat, lon],
                radius=6, color=color, fill=True, fill_opacity=0.7,
                popup=f"Thời gian: {row.get('Ngày - giờ', 'N/A')}<br>Cấp: {row.get('cường độ (cấp BF)', 'N/A')}"
            ).add_to(m)

    if len(points) > 1:
        folium.PolyLine(points, color="blue", weight=2).add_to(m)

    # Hiển thị bản đồ lên web
    st_folium(m, width="100%", height=600)
    st.dataframe(df)
else:
    st.error("Không tìm thấy file dữ liệu besttrack.xlsx")
