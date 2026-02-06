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

warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Hệ thống Giám sát Bão Biển Đông",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    /* Reset lề để map full màn hình */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Sidebar tối màu chuyên nghiệp */
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
    
    /* Style nút bấm */
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
    
    /* Style cho bảng trong Dashboard */
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
    .storm-table tr:last-child td {
        border-bottom: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CÁC HÀM XỬ LÝ DỮ LIỆU ---

@st.cache_data
def load_data(file_path):
    """Đọc và chuẩn hóa dữ liệu bão (Lịch sử & Dự báo)"""
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_excel(file_path)
    
    # Mapping tên cột từ tiếng Việt sang tiếng Anh
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
        "cấp bão": "grade",
        "Thời điểm": "status_raw",   # Cột nhận diện Hiện tại/Dự báo
        "Ngày - giờ": "datetime_str" # Cột thời gian dạng chuỗi
    }
    # Chỉ đổi tên những cột có trong file
    valid_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_rename)
    
    # --- Xử lý Thời gian (DateTime) ---
    # Ưu tiên 1: Cột chuỗi thời gian có sẵn (thường trong file hiện trạng)
    if 'datetime_str' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    
    # Ưu tiên 2: Ghép từ các cột rời (year, mon, day, hour)
    elif all(c in df.columns for c in ['year', 'mon', 'day', 'hour']):
        try:
            time_cols = ['year', 'mon', 'day', 'hour']
            # Ép kiểu số
            for col in time_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=time_cols)
            df[time_cols] = df[time_cols].astype(int)
            
            temp_df = df[time_cols].rename(columns={'mon': 'month'})
            df['dt'] = pd.to_datetime(temp_df)
        except Exception:
            pass
            
    # --- Phân loại Trạng thái (Past / Current / Forecast) ---
    if 'status_raw' in df.columns:
        def categorize(val):
            val_str = str(val).lower()
            if 'dự báo' in val_str: return 'forecast'
            if 'hiện tại' in val_str: return 'current'
            return 'past'
        df['status'] = df['status_raw'].apply(categorize)
    else:
        # Nếu không có cột status, mặc định là lịch sử (past)
        df['status'] = 'past'

    # --- Ép kiểu dữ liệu số ---
    for col in ['lat', 'lon', 'wind_kt', 'pressure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Loại bỏ dữ liệu rác (không có tọa độ hoặc thời gian)
    return df.dropna(subset=['lat', 'lon', 'dt'])

def get_color_by_wind(wind_kt):
    """Màu sắc theo cấp gió (Thang Beaufort/Saffir-Simpson)"""
    if pd.isna(wind_kt): return 'gray'
    w = float(wind_kt)
    if w < 34: return '#00CCFF'  # Áp thấp nhiệt đới (Xanh dương)
    if w < 64: return '#00FF00'  # Bão thường (Xanh lá)
    if w < 83: return '#FFFF00'  # Cấp 1 (Vàng)
    if w < 96: return '#FFAE00'  # Cấp 2 (Cam)
    if w < 113: return '#FF0000' # Cấp 3 (Đỏ)
    if w < 137: return '#FF00FF' # Cấp 4 (Tím)
    return '#800080'             # Cấp 5 (Tím đậm)

# --- 4. TẠO HTML DASHBOARD NỔI ---

def create_dashboard_html(df, selected_storms):
    """Tạo mã HTML cho hộp thông tin nổi bên góc phải"""
    if df.empty or not selected_storms:
        return ""
    
    # Kiểm tra xem dữ liệu có chứa thông tin "Dự báo" hay không
    has_active_data = df['status'].isin(['current', 'forecast']).any()
    
    content_html = ""
    
    if has_active_data:
        # --- CHẾ ĐỘ 1: TIN BÃO (Hiển thị chi tiết từng cơn bão đang hoạt động) ---
        for storm_name in selected_storms:
            sub = df[df['name'] == storm_name].sort_values('dt')
            if sub.empty: continue
            
            # Lấy vị trí hiện tại
            current_pt = sub[sub['status'] == 'current']
            # Nếu không có nhãn 'current', lấy điểm mới nhất trong quá khứ
            if current_pt.empty:
                past_pts = sub[sub['status'] == 'past']
                if not past_pts.empty:
                    current_pt = past_pts.iloc[-1:]
                else:
                    current_pt = sub.iloc[-1:] # Fallback
            
            cur = current_pt.iloc[0]
            
            # Lấy danh sách dự báo
            forecasts = sub[sub['status'] == 'forecast']
            
            forecast_rows = ""
            for _, r in forecasts.iterrows():
                forecast_rows += f"""
                <tr>
                    <td>{r['dt'].strftime('%d/%m %Hh')}</td>
                    <td>{r['lat']}N {r['lon']}E</td>
                    <td>
                        <span style="background:{get_color_by_wind(r.get('wind_kt',0))}; padding:2px 5px; border-radius:3px; color:black; font-weight:bold;">
                            {int(r.get('wind_kt',0))}
                        </span>
                    </td>
                </tr>
                """
            
            content_html += f"""
            <div style="margin-bottom: 15px; border-bottom: 1px solid #ccc; padding-bottom: 10px;">
                <h3 style="margin:0 0 5px 0; color:#d63384; font-size:16px;">BÃO {storm_name.upper()}</h3>
                <div style="background:#f0f2f6; padding:8px; border-radius:5px; font-size:13px; margin-bottom:5px; border-left: 4px solid #007bff;">
                    <b>📍 Vị trí lúc {cur['dt'].strftime('%Hh %d/%m')}</b><br>
                    Tọa độ: {cur['lat']}N - {cur['lon']}E<br>
                    Gió mạnh nhất: <b style="color:red; font-size:14px;">{int(cur.get('wind_kt',0))} kt</b><br>
                    Khí áp: {int(cur.get('pressure',0))} mb
                </div>
                <div style="font-weight:bold; font-size:12px; margin-top:8px;">🔮 DỰ BÁO ĐƯỜNG ĐI:</div>
                <table class="storm-table">
                    <tr><th>Thời gian</th><th>Tọa độ</th><th>Gió (kt)</th></tr>
                    {forecast_rows if forecast_rows else "<tr><td colspan='3'>Chưa có tin dự báo</td></tr>"}
                </table>
            </div>
            """
            
    else:
        # --- CHẾ ĐỘ 2: LỊCH SỬ (Danh sách tóm tắt nhiều cơn bão) ---
        rows = ""
        for storm_name in selected_storms:
            sub = df[df['name'] == storm_name].sort_values('dt', ascending=False)
            if sub.empty: continue
            latest = sub.iloc[0]
            
            rows += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="text-align:left; font-weight:bold; color:#007bff;">{storm_name}</td>
                <td>{latest['dt'].strftime('%Y-%m-%d')}</td>
                <td>
                    <span style="background:{get_color_by_wind(latest.get('wind_kt',0))}; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:11px;">
                        {int(latest.get('wind_kt',0))} kt
                    </span>
                </td>
            </tr>
            """
        
        content_html = f"""
        <div style="margin-bottom: 10px;">
            <h4 style="margin: 0 0 10px 0; font-size: 16px; color: #d63384;">🌪️ LỊCH SỬ BÃO ĐÃ LỌC</h4>
            <table class="storm-table">
                <thead>
                    <tr style="background: #f8f9fa;">
                        <th style="color:#333; text-align:left;">Tên bão</th>
                        <th style="color:#333;">Ngày cuối</th>
                        <th style="color:#333;">Cường độ</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    # Legend (Chú giải chung)
    legend_html = """
    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 11px;">
        <div style="margin-bottom:5px;">
            <b>Ký hiệu đường đi:</b><br> 
            <span style="color:black; font-weight:bold">─────</span> Thực tế &nbsp;|&nbsp; 
            <span style="color:black; border-bottom: 2px dashed black;">- - - -</span> Dự báo
        </div>
        <div style="display: flex; gap: 3px; flex-wrap: wrap;">
            <span style="background:#00CCFF; padding:2px 4px; border-radius:3px;">TD</span>
            <span style="background:#00FF00; padding:2px 4px; border-radius:3px;">TS</span>
            <span style="background:#FFFF00; padding:2px 4px; border-radius:3px;">C1</span>
            <span style="background:#FFAE00; padding:2px 4px; border-radius:3px;">C2</span>
            <span style="background:#FF0000; padding:2px 4px; border-radius:3px; color:white;">C3</span>
            <span style="background:#FF00FF; padding:2px 4px; border-radius:3px; color:white;">C4+</span>
        </div>
    </div>
    """

    # Wrapper Box
    full_html = f"""
    <div style="
        position: fixed; 
        top: 20px; 
        right: 20px; 
        width: 320px; 
        max-height: 85vh;
        overflow-y: auto;
        z-index: 99999; 
        background-color: rgba(255, 255, 255, 0.95); 
        border-radius: 8px; 
        padding: 15px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #ccc;
        font-family: Arial, sans-serif;
    ">
        {content_html}
        {legend_html}
    </div>
    """
    return full_html

# --- 5. TẠO ẢNH TĨNH CHẤT LƯỢNG CAO (BACKEND) ---

def generate_static_image(df, selected_storms):
    """Sử dụng Matplotlib & Cartopy để vẽ bản đồ xuất ra file PNG"""
    fig = plt.figure(figsize=(14, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Cấu hình bản đồ nền
    extent = [98, 122, 6, 24] # Khung Biển Đông
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, linestyle="--", edgecolor='gray')
    
    # Lưới kinh vĩ tuyến
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Vẽ từng cơn bão
    for storm_name in selected_storms:
        sub = df[df['name'] == storm_name].sort_values('dt')
        if sub.empty: continue
        
        # Tách phần Quá khứ và Dự báo
        past = sub[sub['status'] != 'forecast']
        forecast = sub[sub['status'] == 'forecast']
        
        # 1. Vẽ đường Quá khứ (Nét liền)
        if not past.empty:
            ax.plot(past['lon'], past['lat'], transform=ccrs.PlateCarree(),
                    color='blue', linewidth=2, label=f"{storm_name} (Thực tế)", zorder=5)
            # Nối điểm cuối quá khứ với điểm đầu dự báo (nếu có) để đường liền mạch
            if not forecast.empty:
                connect_lon = [past.iloc[-1]['lon'], forecast.iloc[0]['lon']]
                connect_lat = [past.iloc[-1]['lat'], forecast.iloc[0]['lat']]
                ax.plot(connect_lon, connect_lat, transform=ccrs.PlateCarree(),
                        color='red', linewidth=2, linestyle='--', zorder=5)

        # 2. Vẽ đường Dự báo (Nét đứt)
        if not forecast.empty:
            ax.plot(forecast['lon'], forecast['lat'], transform=ccrs.PlateCarree(),
                    color='red', linewidth=2, linestyle='--', label=f"{storm_name} (Dự báo)", zorder=5)
            
        # 3. Vẽ các điểm (Markers)
        for _, row in sub.iterrows():
            c = get_color_by_wind(row.get('wind_kt', 0))
            ax.scatter(row['lon'], row['lat'], c=c, s=30, transform=ccrs.PlateCarree(), 
                       edgecolor='black', linewidth=0.5, zorder=6)
            
        # 4. Tên bão (Tại điểm bắt đầu)
        if not sub.empty:
            start = sub.iloc[0]
            ax.text(start['lon'], start['lat'], storm_name, transform=ccrs.PlateCarree(),
                    fontsize=9, weight='bold', color='darkblue', ha='right', va='bottom',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.legend(loc='upper right', fontsize=10)
    ax.set_title("SƠ ĐỒ QUỸ ĐẠO BÃO", fontsize=15, weight='bold', pad=15)
    
    # Lưu vào buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    plt.close(fig)
    return buf

# --- 6. GIAO DIỆN CHÍNH (MAIN APP) ---

def main():
    # --- SIDEBAR: CONTROL PANEL ---
    with st.sidebar:
        st.title("🌪️ CONTROL PANEL")
        st.markdown("---")
        
        # 1. Upload File
        default_file = "besttrack_capgio.xlsx"
        uploaded_file = st.file_uploader("Tải lên file dữ liệu (.xlsx)", type=["xlsx"])
        
        data_source = None
        if uploaded_file:
            data_source = uploaded_file
        elif os.path.exists(default_file):
            data_source = default_file
        
        if not data_source:
            st.warning("⚠️ Vui lòng tải file dữ liệu!")
            st.stop()
            
        df = load_data(data_source)
        if df is None or df.empty:
            st.error("❌ Không đọc được dữ liệu hoặc file rỗng.")
            st.stop()
            
        # 2. Bộ lọc (Filters)
        st.subheader("🛠️ Bộ lọc hiển thị")
        
        # Lọc Năm
        if 'year' in df.columns:
            all_years = sorted(df['year'].unique())
            selected_years = st.multiselect("Năm:", all_years, default=all_years[-1:] if all_years else None)
        else:
            selected_years = [] # File current có thể không có cột year rõ ràng
            
        # Lọc Bão
        if selected_years:
            temp_df = df[df['year'].isin(selected_years)]
        else:
            temp_df = df
            
        all_storms = temp_df['name'].unique()
        selected_storms_names = st.multiselect("Chọn cơn bão:", all_storms, default=all_storms)
        
        # Lọc Gió
        if not temp_df.empty and 'wind_kt' in temp_df.columns:
            min_w = int(temp_df['wind_kt'].min())
            max_w = int(temp_df['wind_kt'].max())
            if min_w < max_w:
                wind_range = st.slider("Cường độ gió (kt):", min_w, max_w, (min_w, max_w))
            else:
                wind_range = (min_w, max_w)
        else:
            wind_range = (0, 200)

        # ÁP DỤNG LỌC
        final_df = temp_df[
            (temp_df['name'].isin(selected_storms_names)) &
            (temp_df['wind_kt'] >= wind_range[0]) &
            (temp_df['wind_kt'] <= wind_range[1])
        ]
        
        st.success(f"Hiển thị: {len(final_df)} điểm dữ liệu.")
        
        st.markdown("---")
        
        # 3. Khu vực Download
        st.subheader("📥 Xuất dữ liệu")
        if not final_df.empty:
            # Excel
            towrite = io.BytesIO()
            final_df.to_excel(towrite, index=False)
            towrite.seek(0)
            st.download_button("📄 Tải Excel", towrite, "storm_data.xlsx")
            
            # PNG Image
            if st.button("🖼️ Tạo ảnh bản đồ (PNG)"):
                with st.spinner("Đang vẽ bản đồ chất lượng cao..."):
                    img_buf = generate_static_image(final_df, selected_storms_names)
                    st.download_button("⬇️ Tải ảnh xuống", img_buf, "storm_map.png", "image/png")

    # --- MAIN MAP AREA ---
    
    # Khởi tạo bản đồ Folium
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles="CartoDB positron")
    
    # Layer Group
    fg = folium.FeatureGroup(name="Bão")
    
    if not final_df.empty:
        for storm_name in selected_storms_names:
            storm_data = final_df[final_df['name'] == storm_name].sort_values('dt')
            if storm_data.empty: continue
            
            # Tách dữ liệu để vẽ nét liền (quá khứ) và nét đứt (dự báo)
            past_data = storm_data[storm_data['status'] != 'forecast']
            forecast_data = storm_data[storm_data['status'] == 'forecast']
            
            # 1. Vẽ đường quá khứ
            if not past_data.empty:
                coords = past_data[['lat', 'lon']].values.tolist()
                folium.PolyLine(
                    locations=coords, color='black', weight=2, opacity=0.7,
                    tooltip=f"{storm_name} (Thực tế)"
                ).add_to(fg)
                
            # 2. Vẽ đường dự báo (Dashed)
            if not forecast_data.empty:
                # Nối điểm cuối quá khứ với điểm đầu dự báo
                if not past_data.empty:
                    last_past = past_data.iloc[-1]
                    first_fc = forecast_data.iloc[0]
                    conn_coords = [[last_past['lat'], last_past['lon']], [first_fc['lat'], first_fc['lon']]]
                    folium.PolyLine(locations=conn_coords, color='red', weight=2, dash_array='5, 5', opacity=0.7).add_to(fg)
                
                coords_fc = forecast_data[['lat', 'lon']].values.tolist()
                folium.PolyLine(
                    locations=coords_fc, color='red', weight=2, dash_array='5, 5', opacity=0.7,
                    tooltip=f"{storm_name} (Dự báo)"
                ).add_to(fg)

            # 3. Vẽ các điểm Marker
            for _, row in storm_data.iterrows():
                # Nội dung Popup
                status_txt = "DỰ BÁO" if row['status'] == 'forecast' else "THỰC TẾ"
                popup_html = f"""
                <div style="font-family:Arial; width:200px; font-size:12px;">
                    <b>{row['name']} ({status_txt})</b><br>
                    Thời gian: {row['dt'].strftime('%d/%m %Hh')}<br>
                    Vị trí: {row['lat']}N - {row['lon']}E<br>
                    Gió: {int(row.get('wind_kt', 0))} kt (Cấp {int(row.get('grade',0)) if pd.notna(row.get('grade')) else '-'})<br>
                    Áp suất: {row.get('pressure', 'N/A')} mb
                </div>
                """
                
                color = get_color_by_wind(row.get('wind_kt', 0))
                
                # Marker: Hình tròn nhỏ
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5 if row['status'] != 'current' else 8, # Điểm hiện tại to hơn
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8 if row['status'] != 'forecast' else 0.5,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(fg)
                
                # Nếu là điểm hiện tại -> Thêm hiệu ứng Pulse hoặc viền đỏ
                if row['status'] == 'current':
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=12, color='red', fill=False, weight=1
                    ).add_to(fg)

    fg.add_to(m)
    
    # Vẽ lưới kinh vĩ tuyến (Thủ công)
    for lon in range(100, 131, 5):
        folium.PolyLine([[0, lon], [35, lon]], color='gray', weight=0.5, opacity=0.3, dash_array='5').add_to(m)
    for lat in range(0, 36, 5):
        folium.PolyLine([[lat, 90], [lat, 140]], color='gray', weight=0.5, opacity=0.3, dash_array='5').add_to(m)

    # --- INJECT DASHBOARD HTML ---
    if not final_df.empty:
        dashboard_html = create_dashboard_html(final_df, selected_storms_names)
        st.markdown(dashboard_html, unsafe_allow_html=True)

    # Hiển thị bản đồ
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
