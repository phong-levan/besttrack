import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Hệ thống theo dõi xoáy thuận nhiệt đới", layout="wide")
st.title("🌀 Bản đồ theo dõi xoáy thuận nhiệt đới")

# 1. Đọc dữ liệu từ file Excel của bạn
FILE_PATH = "besttrack.xlsx"

@st.cache_data
def load_data():
    if os.path.exists(FILE_PATH):
        df = pd.read_excel(FILE_PATH)
        # Chuyển đổi cột ngày giờ sang dạng chuỗi để hiển thị
        if 'Ngày - giờ' in df.columns:
            df['Ngày - giờ'] = df['Ngày - giờ'].astype(str)
        return df
    return None

df = load_data()

if df is not None:
    # 2. Khởi tạo bản đồ Folium (Cho phép thu phóng)
    # Tọa độ trung tâm Biển Đông
    m = folium.Map(location=[15.8, 110.0], zoom_start=5, tiles="CartoDB positron")

    # 3. Vẽ quỹ đạo và các điểm tâm bão
    points = []
    for i, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        if pd.notna(lat) and pd.notna(lon):
            points.append([lat, lon])
            
            # Xác định màu sắc dựa trên trạng thái (Quá khứ hay Dự báo)
            is_past = "quá khứ" in str(row.get('Thời điểm', '')).lower()
            color = "black" if is_past else "red"
            
            # Tạo nội dung khi nhấn vào điểm bão
            popup_text = f"""
            <b>Thời gian:</b> {row.get('Ngày - giờ', 'N/A')}<br>
            <b>Cường độ:</b> Cấp {row.get('cường độ (cấp BF)', 'N/A')}<br>
            <b>Áp suất:</b> {row.get('Pmin (mb)', 'N/A')} mb<br>
            <b>Vận tốc:</b> {row.get('Vmax (km/h)', 'N/A')} km/h
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).addTo(m)

    # Vẽ đường nối quỹ đạo bão
    if len(points) > 1:
        folium.PolyLine(points, color="blue", weight=2.5, opacity=0.7).addTo(m)

    # 4. Hiển thị bản đồ lên Streamlit
    st_folium(m, width="100%", height=600)

    # 5. Hiển thị bảng dữ liệu chi tiết bên dưới
    with st.expander("Xem bảng dữ liệu chi tiết"):
        st.dataframe(df)

else:

    st.error(f"Không tìm thấy file {FILE_PATH}. Vui lòng kiểm tra lại thư mục dự án.")

