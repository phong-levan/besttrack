# -*- coding: utf-8 -*-
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
import warnings

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH HỆ THỐNG (ẨN SIDEBAR MẶC ĐỊNH) ---
st.set_page_config(
    page_title="Hệ thống Giám sát Bão",
    layout="wide",
    initial_sidebar_state="collapsed" # <-- Quan trọng: Thu gọn thanh bên
)

# --- 2. CSS TỐI GIẢN (CHỈ GIỮ FULL SCREEN) ---
st.markdown("""
    <style>
    /* Reset lề để map full màn hình */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Ẩn Header/Footer/Hamburger menu mặc định của Streamlit cho sạch mắt */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Tùy chỉnh giao diện Sidebar khi mở ra */
    [data-testid="stSidebar"] {
        background-color: #1c2331;
        color: white;
        opacity: 0.9; /* Hơi trong suốt để đẹp hơn */
    }
    [data-testid="stSidebar"] h1, h2, h3 { color: #00d4ff !important; }
    .stMarkdown, .stText, label { color: #e0e0e0 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_excel(file_path)
    
    # Mapping
    rename_map = {
        "tên bão": "name", "biển đông": "storm_no", 
        "năm": "year", "tháng": "mon", "ngày": "day", "giờ": "hour", 
        "vĩ độ": "lat", "kinh độ": "lon", 
        "gió (kt)": "wind_kt", "khí áp (mb)": "pressure", 
        "Thời điểm": "status_raw", "Ngày - giờ": "datetime_str"
    }
    valid_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_rename)
    
    # Xử lý thời gian
    if 'datetime_str' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    elif all(c in df.columns for c in ['year', 'mon', 'day', 'hour']):
        try:
            time_cols = ['year', 'mon', 'day', 'hour']
            for col in time_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=time_cols)
            df[time_cols] = df[time_cols].astype(int)
            temp_df = df[time_cols].rename(columns={'mon': 'month'})
            df['dt'] = pd.to_datetime(temp_df)
        except: pass
            
    # Phân loại trạng thái
    if 'status_raw' in df.columns:
        def categorize(val):
            val_str = str(val).lower()
            if 'dự báo' in val_str: return 'forecast'
            if 'hiện tại' in val_str: return 'current'
            return 'past'
        df['status'] = df['status_raw'].apply(categorize)
    else:
        df['status'] = 'past'

    for col in ['lat', 'lon', 'wind_kt']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df.dropna(subset=['lat', 'lon', 'dt'])

def get_color_by_wind(wind_kt):
    if pd.isna(wind_kt): return 'gray'
    w = float(wind_kt)
    if w < 34: return '#00CCFF'
    if w < 64: return '#00FF00'
    if w < 83: return '#FFFF00'
    if w < 96: return '#FFAE00'
    if w < 113: return '#FF0000'
    if w < 137: return '#FF00FF'
    return '#800080'

# --- 4. MAIN ---

def main():
    # --- SIDEBAR (Giữ lại để lọc dữ liệu, nhưng mặc định bị ẩn) ---
    with st.sidebar:
        st.title("⚙️ CẤU HÌNH")
        default_file = "besttrack_capgio.xlsx"
        uploaded_file = st.file_uploader("File dữ liệu (.xlsx)", type=["xlsx"])
        data_source = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)
        
        if not data_source: st.stop()
        df = load_data(data_source)
        if df is None or df.empty: st.stop()
            
        # Bộ lọc nhanh
        st.subheader("Lọc hiển thị")
        sel_storms = st.multiselect("Chọn Bão:", df['name'].unique(), default=df['name'].unique())
        
        if not df.empty and 'wind_kt' in df.columns:
            min_w, max_w = int(df['wind_kt'].min()), int(df['wind_kt'].max())
            w_range = st.slider("Cấp gió (kt):", min_w, max_w, (min_w, max_w))
            final_df = df[(df['name'].isin(sel_storms)) & (df['wind_kt'] >= w_range[0]) & (df['wind_kt'] <= w_range[1])]
        else: final_df = df

    # --- BẢN ĐỒ ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None)
    
    # 1. Nền bản đồ
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    # 2. Các lớp dữ liệu
    fg_past = folium.FeatureGroup(name="📉 Đường thực tế")
    fg_forecast = folium.FeatureGroup(name="🔮 Đường dự báo")
    fg_points = folium.FeatureGroup(name="📍 Điểm chi tiết")
    
    if not final_df.empty:
        for storm_name in sel_storms:
            storm_data = final_df[final_df['name'] == storm_name].sort_values('dt')
            if storm_data.empty: continue
            
            past_data = storm_data[storm_data['status'] != 'forecast']
            forecast_data = storm_data[storm_data['status'] == 'forecast']
            
            # Vẽ đường thực tế
            if not past_data.empty:
                folium.PolyLine(
                    past_data[['lat', 'lon']].values.tolist(), 
                    color='black', weight=2, opacity=0.8, tooltip=f"{storm_name}"
                ).add_to(fg_past)
                
            # Vẽ đường dự báo
            if not forecast_data.empty:
                # Nối nét
                if not past_data.empty:
                    conn = [[past_data.iloc[-1]['lat'], past_data.iloc[-1]['lon']], 
                            [forecast_data.iloc[0]['lat'], forecast_data.iloc[0]['lon']]]
                    folium.PolyLine(conn, color='red', weight=2, dash_array='5, 5').add_to(fg_forecast)
                
                folium.PolyLine(
                    forecast_data[['lat', 'lon']].values.tolist(), 
                    color='red', weight=2, dash_array='5, 5'
                ).add_to(fg_forecast)

            # Vẽ điểm marker
            for _, row in storm_data.iterrows():
                color = get_color_by_wind(row.get('wind_kt', 0))
                popup = f"<b>{row['name']}</b><br>{row['dt'].strftime('%d/%m %Hh')}<br>{int(row.get('wind_kt',0))}kt"
                
                target_group = fg_forecast if row['status'] == 'forecast' else fg_points
                
                folium.CircleMarker(
                    [row['lat'], row['lon']], radius=5 if row['status'] != 'current' else 9,
                    color=color, fill=True, fill_color=color, fill_opacity=1, popup=popup
                ).add_to(target_group)
                
                if row['status'] == 'current':
                    folium.CircleMarker([row['lat'], row['lon']], radius=14, color='red', fill=False).add_to(fg_points)

    fg_past.add_to(m)
    fg_forecast.add_to(m)
    fg_points.add_to(m)
    
    # Layer Control (Gọn nhẹ)
    folium.LayerControl(collapsed=True).add_to(m)
    
    # Hiển thị Map Full
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
