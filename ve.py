# -*- coding: utf-8 -*-
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patheffects as path_effects
import numpy as np

# --- 1. CẤU HÌNH TRANG WEB (PHẢI Ở DÒNG ĐẦU TIÊN) ---
st.set_page_config(
    page_title="Hệ thống Giám sát Bão Biển Đông",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TÙY CHỈNH: GIAO DIỆN TRÀN VIỀN (IWEATHER STYLE) ---
st.markdown("""
    <style>
    /* 1. Xóa padding mặc định để map full màn hình */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    
    /* 2. Tùy chỉnh Sidebar tối màu */
    [data-testid="stSidebar"] {
        background-color: #1c2331;
        color: white;
    }
    [data-testid="stSidebar"] h1, h2, h3 {
        color: #00d4ff !important;
    }
    .stMarkdown, .stText, label {
        color: #e0e0e0 !important;
    }
    
    /* 3. Ẩn Header/Footer mặc định */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 4. Tinh chỉnh nút bấm */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CÁC HÀM XỬ LÝ DỮ LIỆU ---

@st.cache_data
def load_data(file_path):
    """Đọc dữ liệu từ Excel và chuẩn hóa tên cột (Đã sửa lỗi ngày tháng)"""
    if not os.path.exists(file_path):
        return None
    
    # Đọc file
    df = pd.read_excel(file_path)
    
    # Mapping tên cột (Tiếng Việt -> Tiếng Anh)
    rename_map = {
        "tên bão": "name",
        "biển đông": "storm_no",
        "năm": "year",
        "tháng": "mon",
        "ngày": "day",
        "giờ": "hour",
        "vĩ độ": "lat",
        "kinh độ": "lon",
        "gió (kt)": "wind_kt",
        "khí áp (mb)": "pressure",
        "cấp bão": "grade"
    }
    # Chỉ đổi tên những cột thực sự tồn tại
    valid_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_rename)
    
    # --- XỬ LÝ THỜI GIAN AN TOÀN ---
    time_cols = ['year', 'mon', 'day', 'hour']
    
    # Trường hợp 1: File có cột tách rời (year, mon, day...)
    if all(c in df.columns for c in time_cols):
        # Ép kiểu số, lỗi thành NaN để tránh crash
        for col in time_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Loại bỏ dòng lỗi
        df = df.dropna(subset=time_cols)
        
        # Chuyển float (2024.0) về int (2024)
        df[time_cols] = df[time_cols].astype(int)
        
        # Đổi tên 'mon' -> 'month' cho hàm pd.to_datetime hiểu
        temp_df = df[time_cols].rename(columns={'mon': 'month'})
        
        # Tạo cột dt
        df['dt'] = pd.to_datetime(temp_df)
    
    # Trường hợp 2: File có cột gộp sẵn (ngay_gio)
    elif 'ngay_gio' in df.columns:
         df['dt'] = pd.to_datetime(df['ngay_gio'], errors='coerce')
    
    # Ép kiểu số cho dữ liệu bão
    for col in ['lat', 'lon', 'wind_kt', 'pressure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Loại bỏ các dòng không có tọa độ
    return df.dropna(subset=['lat', 'lon', 'dt'])

def get_color_by_wind(wind_kt):
    """Màu sắc đường đi bão dựa trên sức gió (kt)"""
    if pd.isna(wind_kt): return 'gray'
    if wind_kt < 34: return '#00CCFF'  # TD (Xanh dương nhạt)
    if wind_kt < 64: return '#00FF00'  # TS (Xanh lá)
    if wind_kt < 83: return '#FFFF00'  # Cat 1 (Vàng)
    if wind_kt < 96: return '#FFAE00'  # Cat 2 (Cam)
    if wind_kt < 113: return '#FF0000' # Cat 3 (Đỏ)
    if wind_kt < 137: return '#FF00FF' # Cat 4 (Tím)
    return '#800080'                   # Cat 5 (Tím đậm)

# --- 4. ENGINE TẠO ẢNH TĨNH (MATPLOTLIB + CARTOPY) ---
def generate_static_image(df, selected_storms, show_labels=True):
    """Tạo ảnh PNG chất lượng cao để tải về"""
    fig = plt.figure(figsize=(12, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Phạm vi Biển Đông
    extent = [98, 125, 5, 25]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Nền bản đồ
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, linestyle="--", edgecolor='gray')
    
    # Gridlines
    xticks = np.arange(100, 126, 5)
    yticks = np.arange(5, 26, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    # Vẽ bão
    for storm_name in selected_storms:
        sub = df[df['name'] == storm_name].sort_values('dt')
        if sub.empty: continue
        
        # Vẽ đường đi
        ax.plot(sub['lon'], sub['lat'], transform=ccrs.PlateCarree(), 
                linewidth=2, label=storm_name, zorder=5)
        
        # Vẽ điểm
        ax.scatter(sub['lon'], sub['lat'], c='red', s=15, transform=ccrs.PlateCarree(), zorder=6)
        
        # Tên bão
        if show_labels:
            start_pt = sub.iloc[0]
            ax.text(start_pt['lon'], start_pt['lat'], storm_name,
                    transform=ccrs.PlateCarree(), fontsize=8, weight='bold', color='blue',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.legend(loc='upper right', title="Danh sách bão")
    ax.set_title(f"SƠ ĐỒ QUỸ ĐẠO BÃO", fontsize=14, weight='bold')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    plt.close(fig)
    return buf

# --- 5. GIAO DIỆN CHÍNH ---

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🌪️ CONTROL PANEL")
        st.markdown("---")
        
        # 1. Upload/Chọn file
        # Lưu ý: Trên Streamlit Cloud, file mặc định phải nằm cùng thư mục git
        default_file = "besttrack_capgio.xlsx"
        
        uploaded_file = st.file_uploader("Tải lên file dữ liệu (xlsx)", type=["xlsx"])
        
        if uploaded_file:
            data_source = uploaded_file
        elif os.path.exists(default_file):
            data_source = default_file
        else:
            st.warning("⚠️ Chưa có dữ liệu! Vui lòng tải file excel lên.")
            st.stop()
            
        df = load_data(data_source)
        
        if df is None or df.empty:
            st.error("File dữ liệu rỗng hoặc sai định dạng!")
            st.stop()
            
        # 2. Bộ lọc
        st.subheader("🛠️ Bộ lọc dữ liệu")
        
        # Lọc Năm
        all_years = sorted(df['year'].unique())
        selected_years = st.multiselect("Chọn Năm:", all_years, default=all_years[-1:] if all_years else None)
        
        # Lọc Tháng
        all_months = sorted(df['mon'].unique())
        selected_months = st.multiselect("Chọn Tháng:", all_months, default=all_months)
        
        # Áp dụng lọc sơ bộ
        temp_df = df[df['year'].isin(selected_years) & df['mon'].isin(selected_months)]
        
        # Lọc Tên Bão
        all_storms = temp_df['name'].unique()
        selected_storms_names = st.multiselect("Chọn Bão:", all_storms, default=all_storms)
        
        # Lọc Cấp Gió
        if not temp_df.empty:
            min_wind, max_wind = int(temp_df['wind_kt'].min()), int(temp_df['wind_kt'].max())
            wind_range = st.slider("Phạm vi sức gió (kt):", min_wind, max_wind, (min_wind, max_wind))
            
            final_df = temp_df[
                (temp_df['name'].isin(selected_storms_names)) &
                (temp_df['wind_kt'] >= wind_range[0]) &
                (temp_df['wind_kt'] <= wind_range[1])
            ]
        else:
            final_df = temp_df

        st.success(f"Hiển thị: {len(final_df)} điểm / {len(selected_storms_names)} cơn bão.")
        
        st.markdown("---")
        # 3. Download
        st.subheader("📥 Xuất dữ liệu")
        
        if not final_df.empty:
            # Excel
            towrite = io.BytesIO()
            final_df.to_excel(towrite, index=False, engine='xlsxwriter')
            towrite.seek(0)
            st.download_button(label="📄 Tải dữ liệu (Excel)", data=towrite, file_name="storm_data.xlsx")
            
            # Image PNG
            if st.button("🖼️ Tạo ảnh bản đồ (PNG)"):
                with st.spinner("Đang vẽ bản đồ chất lượng cao..."):
                    img_buf = generate_static_image(final_df, selected_storms_names)
                    st.download_button(
                        label="⬇️ Tải ảnh xuống",
                        data=img_buf,
                        file_name="storm_map_hd.png",
                        mime="image/png"
                    )

    # --- MAIN MAP ---
    m = folium.Map(location=[16.0, 112.0], zoom_start=6, tiles="CartoDB positron") 
    
    feature_group = folium.FeatureGroup(name="Đường đi bão")
    
    if not final_df.empty:
        for storm_name in selected_storms_names:
            storm_data = final_df[final_df['name'] == storm_name].sort_values('dt')
            if storm_data.empty: continue
            
            # Vẽ đường
            coordinates = storm_data[['lat', 'lon']].values.tolist()
            folium.PolyLine(
                locations=coordinates, color="black", weight=2, opacity=0.6,
                tooltip=f"Đường đi: {storm_name}"
            ).add_to(feature_group)
            
            # Vẽ điểm
            for _, row in storm_data.iterrows():
                popup_content = f"""
                <div style='font-family:Arial; font-size:12px; width:200px'>
                    <b>{row['name']}</b> ({row['dt'].strftime('%d/%m %Hh')})<br>
                    Gió: {row['wind_kt']} kt<br>
                    Áp suất: {row.get('pressure', 'N/A')} mb
                </div>
                """
                color = get_color_by_wind(row.get('wind_kt', 0))
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5, color=color, fill=True, fill_color=color, fill_opacity=1.0,
                    popup=folium.Popup(popup_content, max_width=250)
                ).add_to(feature_group)

    feature_group.add_to(m)
    
    # Lưới kinh vĩ tuyến
    for lon in range(100, 126, 5):
        folium.PolyLine([[0, lon], [30, lon]], color='gray', weight=0.5, opacity=0.3, dash_array='5').add_to(m)
    for lat in range(0, 31, 5):
        folium.PolyLine([[lat, 95], [lat, 130]], color='gray', weight=0.5, opacity=0.3, dash_array='5').add_to(m)

    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
