import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import io
import matplotlib.pyplot as plt
from math import radians, sin, cos, asin, sqrt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- CẤU HÌNH HỆ THỐNG ---
ICON_DIR = "icon"
HISTORY_FILE = "history_tracking.xlsx"
DATA_FILE = "besttrack.xlsx"
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(page_title="Hệ thống Dự báo Bão - Phong Le", layout="wide")

# --- 1. XỬ LÝ DỮ LIỆU & LƯU TRỮ LỊCH SỬ ---
def process_and_log_history(df):
    # Lọc các điểm đã qua (quá khứ) dựa trên cột 'Thời điểm'
    past_df = df[df['Thời điểm'].str.contains("quá khứ", case=False, na=False)].copy()
    
    if os.path.exists(HISTORY_FILE):
        old_history = pd.read_excel(HISTORY_FILE)
        # Gộp và xóa trùng lặp để duy trì bộ dữ liệu Best Track sạch
        new_history = pd.concat([old_history, past_df]).drop_duplicates(subset=['Ngày - giờ'])
        new_history.to_excel(HISTORY_FILE, index=False)
    else:
        past_df.to_excel(HISTORY_FILE, index=False)
    return past_df

# --- 2. THUẬT TOÁN NỘI SUY (DENSIFY - 10KM) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_data(df, step_km=1):
    rows = []
    for i in range(len(df)-1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        n = max(1, int(np.ceil(dist/step_km)))
        for j in range(n):
            f = j/n
            rows.append({
                'lat': p1['lat'] + (p2['lat']-p1['lat'])*f,
                'lon': p1['lon'] + (p2['lon']-p1['lon'])*f,
                'r6': p1.get('bán kính gió mạnh cấp 6 (km)',0)*(1-f) + p2.get('bán kính gió mạnh cấp 6 (km)',0)*f,
                'r10': p1.get('bán kính gió mạnh cấp 10 (km)',0)*(1-f) + p2.get('bán kính gió mạnh cấp 10 (km)',0)*f,
                'rc': p1.get('bán kính tâm (km)',0)*(1-f) + p2.get('bán kính tâm (km)',0)*f
            })
    rows.append(df.iloc[-1].to_dict())
    return pd.DataFrame(rows)

# --- 3. XUẤT ẢNH PNG (CHỨA ĐỦ KINH VĨ ĐỘ & BẢNG) ---
def export_static_png(df):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    # Vẽ quỹ đạo cơ bản
    ax.plot(df['lon'], df['lat'], 'k-o', markersize=3, linewidth=1)
    ax.set_xlabel("Kinh độ (E)")
    ax.set_ylabel("Vĩ độ (N)")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Chèn bảng thông tin vào góc ảnh
    table_data = df[['Ngày - giờ', 'lat', 'lon', 'cường độ (cấp BF)']].tail(5).values
    table = ax.table(cellText=table_data, colLabels=['Thời gian', 'Vĩ độ', 'Kinh độ', 'Cấp'], 
                     loc='upper right', bbox=[0.6, 0.7, 0.38, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# --- GIAO DIỆN CHÍNH ---
if os.path.exists(DATA_FILE):
    raw_df = pd.read_excel(DATA_FILE)
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])
    
    # Tự động lưu lịch sử mỗi khi chạy
    past_positions = process_and_log_history(raw_df)

    # SIDEBAR: HỘP CÔNG CỤ
    with st.sidebar:
        st.header("🛠️ Công cụ Xuất dữ liệu")
        
        # Xuất Excel dự báo
        excel_buf = io.BytesIO()
        raw_df.to_excel(excel_buf, index=False)
        st.download_button("📥 Tải Excel Dự báo", excel_buf.getvalue(), "du_bao_bao.xlsx")
        
        # Xuất Lịch sử đã qua
        if os.path.exists(HISTORY_FILE):
            hist_buf = io.BytesIO()
            pd.read_excel(HISTORY_FILE).to_excel(hist_buf, index=False)
            st.download_button("📜 Tải Lịch sử BestTrack", hist_buf.getvalue(), "history_besttrack.xlsx")

        # Xuất ảnh PNG
        if st.button("🖼️ Khởi tạo ảnh PNG"):
            png_data = export_static_png(raw_df)
            st.download_button("💾 Tải ảnh bản đồ PNG", png_data, "storm_map.png")

    # MAIN CONTENT: BẢN ĐỒ & BẢNG
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        m = folium.Map(location=[16.0, 112.0], zoom_start=5)
        # (Thêm logic vẽ Folium nội suy và Icon như các bước trước)
        st_folium(m, width="100%", height=600)
        
    with col_right:
        st.subheader("📋 Bảng Tin Bão")
        st.image(os.path.join(ICON_DIR, "chuthich.PNG")) # Hiển thị chú thích
        st.table(raw_df[['Ngày - giờ', 'lat', 'lon', 'cường độ (cấp BF)']].tail(8))
else:
    st.error("Thiếu file besttrack.xlsx")
