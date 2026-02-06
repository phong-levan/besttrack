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
import warnings

# Tắt cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(
    page_title="Hệ thống Giám sát Bão Biển Đông",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TÙY CHỈNH (GIAO DIỆN & DASHBOARD) ---
st.markdown("""
    <style>
    /* Reset margin/padding để map full màn hình */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Tùy chỉnh Sidebar giao diện tối (Dark Mode) */
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
    
    /* Ẩn Header/Footer mặc định của Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style cho nút bấm */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        width: 100%;
        border: none;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    
    /* Style cho bảng dữ liệu trong Dashboard */
    .storm-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        color: #333;
        margin-top: 5px;
    }
    .storm-table th {
        background: #007bff;
        color: white;
        padding: 6px;
        text-align: center;
        font-weight: normal;
    }
    .storm-table td {
        border-bottom: 1px solid #ddd;
        padding: 5px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CÁC HÀM XỬ LÝ DỮ LIỆU ---

@st.cache_data
def load_data(file_path):
    """Đọc dữ liệu, chuẩn hóa cột và xử lý lỗi thời gian"""
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_excel(file_path)
    
    # Mapping tên cột (Tiếng Việt -> Tiếng Anh chuẩn code)
    rename_map = {
        "tên bão": "name", "biển đông": "storm_no", 
        "năm": "year", "tháng": "mon", "ngày": "day", "giờ": "hour", 
        "vĩ độ": "lat", "kinh độ": "lon", 
        "gió (kt)": "wind_kt", "khí áp (mb)": "pressure", "cấp bão": "grade",
        "Thời điểm": "status_raw", "Ngày - giờ": "datetime_str"
    }
    valid_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_rename)
    
    # --- XỬ LÝ THỜI GIAN (DATETIME) ---
    # Ưu tiên cột chuỗi thời gian (thường có trong file hiện trạng/dự báo)
    if 'datetime_str' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    
    # Nếu không, ghép từ các cột rời (file lịch sử)
    elif all(c in df.columns for c in ['year', 'mon', 'day', 'hour']):
        try:
            time_cols = ['year', 'mon', 'day', 'hour']
            # Ép kiểu số, biến lỗi thành NaN
            for col in time_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=time_cols)
            df[time_cols] = df[time_cols].astype(int)
            
            temp_df = df[time_cols].rename(columns={'mon': 'month'})
            df['dt'] = pd.to_datetime(temp_df)
        except:
            pass
            
    # --- PHÂN LOẠI TRẠNG THÁI (Thực tế vs Dự báo) ---
    if 'status_raw' in df.columns:
        def categorize(val):
            val_str = str(val).lower()
            if 'dự báo' in val_str: return 'forecast'
            if 'hiện tại' in val_str: return 'current'
            return 'past'
        df['status'] = df['status_raw'].apply(categorize)
    else:
        # Mặc định là quá khứ nếu không có thông tin status
        df['status'] = 'past'

    # Ép kiểu dữ liệu số quan trọng
    for col in ['lat', 'lon', 'wind_kt', 'pressure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Loại bỏ dữ liệu hỏng
    return df.dropna(subset=['lat', 'lon', 'dt'])

def get_color_by_wind(wind_kt):
    """Màu sắc theo cấp gió"""
    if pd.isna(wind_kt): return 'gray'
    w = float(wind_kt)
    if w < 34: return '#00CCFF'  # TD
    if w < 64: return '#00FF00'  # TS
    if w < 83: return '#FFFF00'  # C1
    if w < 96: return '#FFAE00'  # C2
    if w < 113: return '#FF0000' # C3
    if w < 137: return '#FF00FF' # C4
    return '#800080'             # C5

# --- 4. DASHBOARD HTML (RESIZE & COLLAPSE) ---

def create_dashboard_html(df, selected_storms):
    """Tạo HTML cho bảng thông tin nổi: Có tính năng Thu phóng & Cuộn"""
    if df.empty or not selected_storms: return ""
    
    # ID định danh để Javascript thao tác
    box_id = "storm-dashboard-box"
    content_id = "storm-dashboard-content"
    
    has_active_data = df['status'].isin(['current', 'forecast']).any()
    content_html = ""
    
    if has_active_data:
        # --- CHẾ ĐỘ 1: TIN BÃO HIỆN TẠI/DỰ BÁO ---
        for storm_name in selected_storms:
            sub = df[df['name'] == storm_name].sort_values('dt')
            if sub.empty: continue
            
            # Tìm điểm hiện tại
            current_pt = sub[sub['status'] == 'current']
            if current_pt.empty:
                past_pts = sub[sub['status'] == 'past']
                current_pt = past_pts.iloc[-1:] if not past_pts.empty else sub.iloc[-1:]
            cur = current_pt.iloc[0]
            
            # Danh sách điểm dự báo
            forecasts = sub[sub['status'] == 'forecast']
            forecast_rows = ""
            for _, r in forecasts.iterrows():
                forecast_rows += f"""
                <tr>
                    <td>{r['dt'].strftime('%d/%m %Hh')}</td>
                    <td>{r['lat']}N {r['lon']}E</td>
                    <td><span style="background:{get_color_by_wind(r.get('wind_kt',0))}; padding:2px 5px; border-radius:3px; color:black; font-weight:bold;">{int(r.get('wind_kt',0))}</span></td>
                </tr>"""
            
            content_html += f"""
            <div style="margin-bottom: 15px; border-bottom: 1px solid #ccc; padding-bottom: 10px;">
                <h3 style="margin:0 0 5px 0; color:#d63384; font-size:16px;">BÃO {storm_name.upper()}</h3>
                <div style="background:#f0f2f6; padding:8px; border-radius:5px; font-size:13px; margin-bottom:5px; border-left: 4px solid #007bff;">
                    <b>📍 Vị trí lúc {cur['dt'].strftime('%Hh %d/%m')}</b><br>
                    Tọa độ: {cur['lat']}N - {cur['lon']}E<br>
                    Gió: <b style="color:red;">{int(cur.get('wind_kt',0))} kt</b> | P: {int(cur.get('pressure',0))} mb
                </div>
                <div style="font-weight:bold; font-size:12px; margin-top:8px;">🔮 DỰ BÁO:</div>
                <table class="storm-table">
                    <tr><th>Thời gian</th><th>Tọa độ</th><th>Gió</th></tr>
                    {forecast_rows if forecast_rows else "<tr><td colspan='3'>--</td></tr>"}
                </table>
            </div>"""
    else:
        # --- CHẾ ĐỘ 2: LỊCH SỬ BÃO ---
        rows = ""
        for storm_name in selected_storms:
            sub = df[df['name'] == storm_name].sort_values('dt', ascending=False)
            if sub.empty: continue
            latest = sub.iloc[0]
            rows += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="text-align:left; font-weight:bold; color:#007bff;">{storm_name}</td>
                <td>{latest['dt'].strftime('%Y-%m-%d')}</td>
                <td><span style="background:{get_color_by_wind(latest.get('wind_kt',0))}; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:11px;">{int(latest.get('wind_kt',0))} kt</span></td>
            </tr>"""
        
        content_html = f"""
        <h4 style="margin: 0 0 10px 0; font-size: 16px; color: #d63384;">🌪️ DANH SÁCH BÃO</h4>
        <table class="storm-table">
            <thead><tr style="background: #f8f9fa;"><th style="color:#333;">Tên</th><th style="color:#333;">Ngày</th><th style="color:#333;">Gió</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    legend_html = """
    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 11px;">
        <div style="display: flex; gap: 3px; flex-wrap: wrap;">
            <span style="background:#00CCFF; padding:2px 4px; border-radius:3px;">TD</span>
            <span style="background:#00FF00; padding:2px 4px; border-radius:3px;">TS</span>
            <span style="background:#FFFF00; padding:2px 4px; border-radius:3px;">C1</span>
            <span style="background:#FFAE00; padding:2px 4px; border-radius:3px;">C2</span>
            <span style="background:#FF0000; padding:2px 4px; border-radius:3px; color:white;">C3</span>
            <span style="background:#FF00FF; padding:2px 4px; border-radius:3px; color:white;">C4+</span>
        </div>
    </div>"""

    # --- HTML + JS (RESIZE LOGIC) ---
    full_html = f"""
    <div id="{box_id}" style="
        position: fixed; 
        top: 20px; 
        right: 20px; 
        width: 320px; 
        min-width: 250px;
        max-width: 90vw;
        min-height: 40px;
        max-height: 90vh;
        z-index: 99999; 
        background-color: rgba(255, 255, 255, 0.95); 
        border-radius: 8px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); 
        border: 1px solid #ccc; 
        font-family: Arial, sans-serif;
        
        /* Kích hoạt tính năng kéo giãn (Resize) */
        resize: both; 
        overflow: auto; 
        transition: background-color 0.3s;
    ">
        <div style="
            background: #007bff; color: white; padding: 8px 15px; 
            border-radius: 8px 8px 0 0;
            display: flex; justify-content: space-between; align-items: center; 
            cursor: pointer; position: sticky; top: 0; z-index: 1000;
        " onclick="toggleDashboard()">
            <span style="font-weight: bold; font-size: 14px;">🛠️ BẢNG THÔNG TIN</span>
            <span id="toggle-icon" style="font-weight: bold; font-size: 18px;">−</span>
        </div>

        <div id="{content_id}" style="padding: 15px;">
            {content_html}
            {legend_html}
            <div style="margin-top:10px; text-align:right; font-size:10px; color:#999;">
                ◢ Kéo góc để chỉnh kích thước
            </div>
        </div>
    </div>

    <script>
        function toggleDashboard() {{
            var content = document.getElementById('{content_id}');
            var icon = document.getElementById('toggle-icon');
            var box = document.getElementById('{box_id}');
            
            if (content.style.display === 'none') {{
                // Mở rộng
                content.style.display = 'block';
                icon.innerHTML = '−';
                box.style.resize = 'both'; // Bật lại resize
                box.style.height = 'auto'; 
            }} else {{
                // Thu gọn
                content.style.display = 'none';
                icon.innerHTML = '+';
                box.style.resize = 'none'; // Khóa resize khi thu gọn
                box.style.width = '200px'; 
                box.style.height = '40px';
            }}
        }}
    </script>
    """
    return full_html

# --- 5. TẠO ẢNH BẢN ĐỒ (BACKEND CARTOPY) ---

def generate_static_image(df, selected_storms):
    """Vẽ bản đồ tĩnh chất lượng cao (PNG)"""
    fig = plt.figure(figsize=(14, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Cấu hình bản đồ nền
    ax.set_extent([98, 122, 6, 24], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, linestyle="--", edgecolor='gray')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    for storm_name in selected_storms:
        sub = df[df['name'] == storm_name].sort_values('dt')
        if sub.empty: continue
        
        # Tách dữ liệu
        past = sub[sub['status'] != 'forecast']
        forecast = sub[sub['status'] == 'forecast']
        
        # Vẽ đường thực tế (Xanh/Đen)
        if not past.empty:
            ax.plot(past['lon'], past['lat'], transform=ccrs.PlateCarree(), 
                    color='blue', linewidth=2, label=f"{storm_name}", zorder=5)
            # Nối nét với dự báo
            if not forecast.empty:
                conn_x = [past.iloc[-1]['lon'], forecast.iloc[0]['lon']]
                conn_y = [past.iloc[-1]['lat'], forecast.iloc[0]['lat']]
                ax.plot(conn_x, conn_y, transform=ccrs.PlateCarree(), color='red', linewidth=2, linestyle='--', zorder=5)

        # Vẽ đường dự báo (Đỏ - Nét đứt)
        if not forecast.empty:
            ax.plot(forecast['lon'], forecast['lat'], transform=ccrs.PlateCarree(), 
                    color='red', linewidth=2, linestyle='--', zorder=5)
            
        # Vẽ điểm
        for _, row in sub.iterrows():
            c = get_color_by_wind(row.get('wind_kt', 0))
            ax.scatter(row['lon'], row['lat'], c=c, s=30, transform=ccrs.PlateCarree(), 
                       edgecolor='black', linewidth=0.5, zorder=6)
            
        # Tên bão
        if not sub.empty:
            start = sub.iloc[0]
            ax.text(start['lon'], start['lat'], storm_name, transform=ccrs.PlateCarree(), 
                    fontsize=9, weight='bold', color='darkblue', 
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.legend(loc='upper right')
    ax.set_title("SƠ ĐỒ QUỸ ĐẠO BÃO", fontsize=15, weight='bold')
    
    # Lưu vào bộ nhớ
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    plt.close(fig)
    return buf

# --- 6. CHƯƠNG TRÌNH CHÍNH (MAIN) ---

def main():
    # --- SIDEBAR: CONTROL PANEL ---
    with st.sidebar:
        st.title("🌪️ CONTROL PANEL")
        st.markdown("---")
        
        # Upload File
        default_file = "besttrack_capgio.xlsx"
        uploaded_file = st.file_uploader("Tải dữ liệu (.xlsx)", type=["xlsx"])
        
        data_source = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)
        
        if not data_source:
            st.warning("⚠️ Vui lòng tải file excel!")
            st.stop()
            
        df = load_data(data_source)
        if df is None or df.empty:
            st.error("❌ Lỗi: File không có dữ liệu hợp lệ.")
            st.stop()
            
        # Bộ Lọc
        st.subheader("🛠️ Bộ lọc")
        
        # Lọc Năm
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            sel_years = st.multiselect("Năm:", years, default=years[-1:])
            temp_df = df[df['year'].isin(sel_years)]
        else:
            temp_df = df
            
        # Lọc Tên Bão
        sel_storms = st.multiselect("Chọn Bão:", temp_df['name'].unique(), default=temp_df['name'].unique())
        
        # Lọc Gió
        if not temp_df.empty and 'wind_kt' in temp_df.columns:
            min_w, max_w = int(temp_df['wind_kt'].min()), int(temp_df['wind_kt'].max())
            w_range = st.slider("Cấp gió (kt):", min_w, max_w, (min_w, max_w))
            final_df = temp_df[
                (temp_df['name'].isin(sel_storms)) & 
                (temp_df['wind_kt'] >= w_range[0]) & 
                (temp_df['wind_kt'] <= w_range[1])
            ]
        else:
            final_df = temp_df
        
        st.success(f"Hiển thị: {len(final_df)} điểm dữ liệu.")
        st.markdown("---")
        
        # Xuất Dữ liệu
        if not final_df.empty:
            st.download_button("📄 Tải Excel", final_df.to_csv(index=False).encode('utf-8'), "storm_data.csv", "text/csv")
            
            if st.button("🖼️ Tạo ảnh PNG (HD)"):
                with st.spinner("Đang vẽ bản đồ chất lượng cao..."):
                    img_buf = generate_static_image(final_df, sel_storms)
                    st.download_button("⬇️ Tải ảnh xuống", img_buf, "map.png", "image/png")

    # --- KHỞI TẠO BẢN ĐỒ ---
    # tiles=None để dùng LayerControl quản lý lớp nền
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None)
    
    # 1. Các lớp nền
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    # 2. Các lớp dữ liệu (FeatureGroups)
    fg_past = folium.FeatureGroup(name="📉 Đường đi Thực tế")
    fg_forecast = folium.FeatureGroup(name="🔮 Đường đi Dự báo")
    fg_points = folium.FeatureGroup(name="📍 Điểm và Nhãn")
    
    if not final_df.empty:
        for storm_name in sel_storms:
            storm_data = final_df[final_df['name'] == storm_name].sort_values('dt')
            if storm_data.empty: continue
            
            past_data = storm_data[storm_data['status'] != 'forecast']
            forecast_data = storm_data[storm_data['status'] == 'forecast']
            
            # Vẽ đường thực tế
            if not past_data.empty:
                coords = past_data[['lat', 'lon']].values.tolist()
                folium.PolyLine(
                    locations=coords, color='black', weight=2, opacity=0.8, 
                    tooltip=f"{storm_name} (Thực tế)"
                ).add_to(fg_past)
                
            # Vẽ đường dự báo
            if not forecast_data.empty:
                # Nối điểm cuối thực tế với điểm đầu dự báo
                if not past_data.empty:
                    conn = [
                        [past_data.iloc[-1]['lat'], past_data.iloc[-1]['lon']], 
                        [forecast_data.iloc[0]['lat'], forecast_data.iloc[0]['lon']]
                    ]
                    folium.PolyLine(locations=conn, color='red', weight=2, dash_array='5, 5').add_to(fg_forecast)
                
                # Vẽ phần dự báo
                fc_coords = forecast_data[['lat', 'lon']].values.tolist()
                folium.PolyLine(
                    locations=fc_coords, color='red', weight=2, dash_array='5, 5', 
                    tooltip=f"{storm_name} (Dự báo)"
                ).add_to(fg_forecast)

            # Vẽ các điểm Marker
            for _, row in storm_data.iterrows():
                color = get_color_by_wind(row.get('wind_kt', 0))
                popup_content = f"""
                <div style='width:180px'>
                    <b>{row['name']}</b><br>
                    Time: {row['dt'].strftime('%d/%m %Hh')}<br>
                    Wind: {int(row.get('wind_kt',0))} kt
                </div>
                """
                
                # Xác định nhóm layer cho điểm
                target_group = fg_forecast if row['status'] == 'forecast' else fg_points
                
                # Điểm tròn nhỏ
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5 if row['status'] != 'current' else 9,
                    color=color, fill=True, fill_color=color, fill_opacity=1,
                    popup=folium.Popup(popup_content, max_width=200)
                ).add_to(target_group)
                
                # Hiệu ứng điểm hiện tại (Vòng tròn đỏ rỗng)
                if row['status'] == 'current':
                    folium.CircleMarker(
                        [row['lat'], row['lon']], radius=14, color='red', fill=False, weight=2
                    ).add_to(fg_points)

    # Thêm các Layer vào bản đồ
    fg_past.add_to(m)
    fg_forecast.add_to(m)
    fg_points.add_to(m)
    
    # 3. Layer Lưới kinh vĩ tuyến (Mặc định ẩn)
    fg_grid = folium.FeatureGroup(name="🌐 Lưới Kinh/Vĩ tuyến", show=False)
    for lon in range(100, 131, 5): 
        folium.PolyLine([[0, lon], [35, lon]], color='gray', weight=0.5, dash_array='5').add_to(fg_grid)
    for lat in range(0, 36, 5): 
        folium.PolyLine([[lat, 90], [lat, 140]], color='gray', weight=0.5, dash_array='5').add_to(fg_grid)
    fg_grid.add_to(m)
    
    # 4. TRÌNH ĐIỀU KHIỂN LAYER (Góc phải trên)
    folium.LayerControl(collapsed=True).add_to(m)

    # 5. INJECT DASHBOARD HTML
    if not final_df.empty:
        st.markdown(create_dashboard_html(final_df, sel_storms), unsafe_allow_html=True)

    # 6. HIỂN THỊ BẢN ĐỒ FULL SCREEN
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
