# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import base64

# --- 1. CẤU HÌNH GIAO DIỆN & CSS TRÀN VIỀN ---
st.set_page_config(page_title="Hệ thống Giám sát Bão Đa tầng", layout="wide")

st.markdown("""
    <style>
    /* Xóa khoảng trắng và thanh cuộn trình duyệt */
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
    # Vẽ lưới kinh vĩ độ mỗi 5 độ
    for lon in range(100, 141, 5):
        folium.PolyLine([[0, lon], [40, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    for lat in range(0, 41, 5):
        folium.PolyLine([[lat, 100], [lat, 140]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    return m

# --- 3. HÀM VẼ CON (DRAWING MODULE) ---
def draw_storm_layer(map_obj, df, layer_name, color, is_past=False):
    """Xử lý vẽ từng lớp dữ liệu bão từ dataframe được cung cấp"""
    fg = folium.FeatureGroup(name=layer_name)
    points = df[['lat', 'lon']].values.tolist()
    
    # Vẽ quỹ đạo bão
    folium.PolyLine(points, color=color, weight=3, opacity=0.7).add_to(fg)
    
    # Vẽ Marker chi tiết cho từng điểm
    for _, row in df.iterrows():
        popup_text = f"Bão: {row.get('Số hiệu', 'N/A')}<br>Cấp gió: {row.get('cường độ (cấp BF)', 0)}<br>Pmin: {row.get('Pmin (mb)', 0)}"
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5 if not is_past else 3,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(fg)
    fg.add_to(map_obj)

# --- 4. CHƯƠNG TRÌNH CHÍNH: NHÚNG VÀ LỌC DỮ LIỆU ---

DATA_FOLDER = "besttrack" # Thư mục chứa các file excel

if os.path.exists(DATA_FOLDER):
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    if files:
        # Bước 1: Khởi tạo bản đồ chung
        m = create_base_map()
        
        # Bước 2: Sidebar - Thanh cuộn tùy chọn Layer
        st.sidebar.title("🛠️ Quản lý Đa tầng Bão")
        
        # Thanh cuộn 1: Chọn file bão hiện tại/dự báo
        st.sidebar.subheader("📍 Trạng thái Hiện tại")
        selected_current = st.sidebar.multiselect("Chọn file bão đang hoạt động:", options=files, default=files[:1])
        
        # Thanh cuộn 2: Lọc dữ liệu bão quá khứ
        st.sidebar.markdown("---")
        st.sidebar.subheader("🕰️ Lọc dữ liệu Quá khứ")
        
        all_data = []
        for f in files:
            path = os.path.join(DATA_FOLDER, f)
            temp_df = pd.read_excel(path, sheet_name='besttrack')
            all_data.append(temp_df)
        
        full_df = pd.concat(all_data).dropna(subset=['lat', 'lon'])
        full_df[['lat', 'lon']] = full_df[['lat', 'lon']].apply(pd.to_numeric)

        # Bộ lọc quá khứ linh hoạt
        storm_list = sorted(full_df['Số hiệu'].unique().tolist())
        sel_storms = st.sidebar.multiselect("Lọc theo Số hiệu bão:", options=storm_list)
        
        bf_range = st.sidebar.slider("Lọc theo Cấp gió (BF):", 0, 18, (0, 18))
        pmin_range = st.sidebar.slider("Lọc theo Khí áp (Pmin):", 900, 1010, (900, 1010))

        # --- Bước 3: Thực hiện vẽ các lớp ---
        
        # Vẽ bão hiện tại (Màu Đỏ)
        for f_name in selected_current:
            path = os.path.join(DATA_FOLDER, f_name)
            df_curr = pd.read_excel(path, sheet_name='besttrack')
            df_curr = df_curr[df_curr['Thời điểm'].str.contains("hiện tại|dự báo", case=False, na=False)]
            if not df_curr.empty:
                draw_storm_layer(m, df_curr, f"Hiện tại: {f_name}", "red")

        # Vẽ bão quá khứ đã lọc (Màu Xanh)
        df_past = full_df[
            (full_df['Thời điểm'].str.contains("quá khứ", case=False, na=False)) &
            (full_df['Số hiệu'].isin(sel_storms)) &
            (full_df['cường độ (cấp BF)'].between(bf_range[0], bf_range[1])) &
            (full_df['Pmin (mb)'].between(pmin_range[0], pmin_range[1]))
        ]
        
        if not df_past.empty:
            draw_storm_layer(m, df_past, "Lớp lọc: Bão quá khứ", "blue", is_past=True)

        # Trình điều khiển Layer trực tiếp trên bản đồ
        folium.LayerControl(position='topleft').add_to(m)

        # Bước 4: Hiển thị Full màn hình
        st_folium(m, width=2000, height=1200, use_container_width=True)
    else:
        st.warning("Thư mục 'besttrack' không có file .xlsx")
else:
    st.error("Không tìm thấy thư mục 'besttrack'")
