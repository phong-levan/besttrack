# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import base64

# --- 1. CẤU HÌNH GIAO DIỆN ---
# Thiết lập tràn viền và ẩn thanh cuộn trình duyệt
st.set_page_config(page_title="Hệ thống Giám sát Bão Đa tầng", layout="wide")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { 
        overflow: hidden !important; height: 100vh; width: 100vw; margin: 0; 
    }
    .main .block-container { padding: 0 !important; max-width: 100% !important; height: 100vh !important; }
    [data-testid="stHeader"], footer { display: none !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; border: none !important; }
    [data-testid="stSidebar"] { z-index: 100; background-color: rgba(248, 249, 250, 0.95); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM BẢN ĐỒ CHUNG (BASE MAP) ---
def create_base_map():
    """Tạo khung bản đồ nền và lưới kinh vĩ độ chuyên dụng"""
    m = folium.Map(location=[17.5, 115.0], zoom_start=6, tiles="OpenStreetMap")
    # Vẽ lưới kinh vĩ độ (mỗi 5 độ)
    for lon in range(100, 141, 5):
        folium.PolyLine([[0, lon], [40, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    for lat in range(0, 41, 5):
        folium.PolyLine([[lat, 100], [lat, 140]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    return m

# --- 3. HÀM CON VẼ DỮ LIỆU (LAYER MODULE) ---
def add_storm_layer(map_obj, df, layer_name, color, is_past=False):
    """Module con xử lý vẽ từng lớp dữ liệu bão"""
    fg = folium.FeatureGroup(name=layer_name)
    
    # Vẽ đường quỹ đạo bão
    points = df[['lat', 'lon']].values.tolist()
    folium.PolyLine(points, color=color, weight=3, opacity=0.7).add_to(fg)
    
    # Vẽ các điểm Marker chi tiết
    for _, row in df.iterrows():
        # Nội dung hiển thị khi nhấp vào điểm
        popup_info = f"""
            <b>Số hiệu:</b> {row.get('Số hiệu', 'N/A')}<br>
            <b>Thời điểm:</b> {row.get('Ngày - giờ', 'N/A')}<br>
            <b>Cấp gió:</b> {row.get('cường độ (cấp BF)', 0)}<br>
            <b>Khí áp:</b> {row.get('Pmin (mb)', 0)} hPa
        """
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5 if not is_past else 3,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_info, max_width=200)
        ).add_to(fg)
        
    fg.add_to(map_obj)

# --- 4. LOGIC XỬ LÝ DỮ LIỆU VÀ GIAO DIỆN ---

# Đường dẫn file Excel (đã chuyển đổi từ file bạn gửi)
EXCEL_FILE = "besttrack_capgio.xlsx" 

if os.path.exists(EXCEL_FILE):
    # Đọc dữ liệu từ sheet 'besttrack'
    df = pd.read_excel(EXCEL_FILE, sheet_name='besttrack')
    df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])

    # Khởi tạo bản đồ chung
    m = create_base_map()

    # --- SIDEBAR: CÁC THANH CUỘN TÙY CHỌN ---
    st.sidebar.title("🛠️ Bảng Điều Khiển")
    
    # Thanh cuộn 1: Bão hiện tại & Dự báo
    st.sidebar.subheader("📍 Trạng thái hiện tại")
    show_current = st.sidebar.toggle("Hiển thị bão Hiện tại/Dự báo", value=True)
    
    if show_current:
        # Lọc dữ liệu có chữ 'hiện tại' hoặc 'dự báo' trong cột Thời điểm
        df_current = df[df['Thời điểm'].str.contains("hiện tại|dự báo", case=False, na=False)]
        if not df_current.empty:
            add_storm_layer(m, df_current, "Lớp: Bão Hiện tại/Dự báo", "red")

    st.sidebar.markdown("---")
    
    # Thanh cuộn 2: Lọc dữ liệu quá khứ
    st.sidebar.subheader("🕰️ Bộ lọc bão quá khứ")
    
    # Lọc theo Số hiệu bão
    storm_list = sorted(df['Số hiệu'].unique().tolist())
    selected_storms = st.sidebar.multiselect("Chọn số hiệu bão:", options=storm_list, default=storm_list[:1])
    
    # Lọc theo Cấp gió (Slider)
    max_bf = int(df['cường độ (cấp BF)'].max())
    bf_range = st.sidebar.slider("Cấp gió (BF):", 0, max_bf, (0, max_bf))
    
    # Lọc theo Khí áp (Slider)
    pmin_min = int(df['Pmin (mb)'].min())
    pmin_max = int(df['Pmin (mb)'].max())
    pmin_range = st.sidebar.slider("Khí áp (Pmin):", pmin_min, pmin_max, (pmin_min, pmin_max))

    # Xử lý lọc dữ liệu quá khứ
    df_past_filtered = df[
        (df['Thời điểm'].str.contains("quá khứ", case=False, na=False)) &
        (df['Số hiệu'].isin(selected_storms)) &
        (df['cường độ (cấp BF)'].between(bf_range[0], bf_range[1])) &
        (df['Pmin (mb)'].between(pmin_range[0], pmin_range[1]))
    ]

    if not df_past_filtered.empty:
        add_storm_layer(m, df_past_filtered, "Lớp: Bão quá khứ (Đã lọc)", "blue", is_past=True)

    # Thêm bảng điều khiển Layer trực tiếp trên bản đồ (Góc trái)
    folium.LayerControl(position='topleft').add_to(m)

    # Đẩy bản đồ ra màn hình (Full width/height)
    st_folium(m, width=2000, height=1200, use_container_width=True)

else:
    st.error(f"Không tìm thấy file {EXCEL_FILE}. Vui lòng kiểm tra lại thư mục.")
