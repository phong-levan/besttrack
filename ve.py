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
    /* Reset lề */
    .block-container { padding: 0 !important; max-width: 100% !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1c2331; color: white; }
    [data-testid="stSidebar"] h1, h2, h3 { color: #00d4ff !important; }
    .stMarkdown, .stText, label { color: #e0e0e0 !important; }
    
    /* Ẩn Header/Footer */
    header {visibility: hidden;} footer {visibility: hidden;}
    
    /* Buttons */
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 5px; width: 100%; border: none; padding: 0.5rem;
    }
    .stButton>button:hover { background-color: #0056b3; }
    
    /* Table Style */
    .storm-table { width: 100%; border-collapse: collapse; font-size: 12px; color: #333; margin-top: 5px; }
    .storm-table th { background: #007bff; color: white; padding: 6px; text-align: center; font-weight: normal; }
    .storm-table td { border-bottom: 1px solid #ddd; padding: 5px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. XỬ LÝ DỮ LIỆU ---

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_excel(file_path)
    
    rename_map = {
        "tên bão": "name", "biển đông": "storm_no", "năm": "year", "tháng": "mon", 
        "ngày": "day", "giờ": "hour", "vĩ độ": "lat", "kinh độ": "lon", 
        "gió (kt)": "wind_kt", "khí áp (mb)": "pressure", "cấp bão": "grade",
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

    for col in ['lat', 'lon', 'wind_kt', 'pressure']:
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

# --- 4. TẠO DASHBOARD THÔNG MINH (CÓ THỂ CUỘN) ---

def create_dashboard_html(df, selected_storms):
    if df.empty or not selected_storms: return ""
    
    # Tạo ID ngẫu nhiên để JS hoạt động độc lập
    box_id = "storm-dashboard-box"
    content_id = "storm-dashboard-content"
    
    has_active_data = df['status'].isin(['current', 'forecast']).any()
    content_html = ""
    
    if has_active_data:
        # CHẾ ĐỘ TIN BÃO (ACTIVE)
        for storm_name in selected_storms:
            sub = df[df['name'] == storm_name].sort_values('dt')
            if sub.empty: continue
            
            current_pt = sub[sub['status'] == 'current']
            if current_pt.empty: current_pt = sub[sub['status'] == 'past'].iloc[-1:] if not sub[sub['status'] == 'past'].empty else sub.iloc[-1:]
            cur = current_pt.iloc[0]
            
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
                    <b>📍 Lúc {cur['dt'].strftime('%Hh %d/%m')}</b><br>
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
        # CHẾ ĐỘ LỊCH SỬ
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

    # --- HTML with JS for Collapsing ---
    full_html = f"""
    <div id="{box_id}" style="
        position: fixed; top: 20px; right: 20px; width: 320px; z-index: 99999; 
        background-color: rgba(255, 255, 255, 0.95); border-radius: 8px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid #ccc; font-family: Arial, sans-serif;
        transition: all 0.3s ease;
    ">
        <div style="
            background: #007bff; color: white; padding: 8px 15px; border-radius: 8px 8px 0 0;
            display: flex; justify-content: space-between; align-items: center; cursor: pointer;
        " onclick="toggleDashboard()">
            <span style="font-weight: bold; font-size: 14px;">BẢNG THÔNG TIN</span>
            <span id="toggle-icon" style="font-weight: bold; font-size: 18px;">−</span>
        </div>

        <div id="{content_id}" style="padding: 15px; max-height: 80vh; overflow-y: auto;">
            {content_html}
            {legend_html}
        </div>
    </div>

    <script>
        function toggleDashboard() {{
            var content = document.getElementById('{content_id}');
            var icon = document.getElementById('toggle-icon');
            var box = document.getElementById('{box_id}');
            
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                icon.innerHTML = '−';
                box.style.width = '320px'; // Khôi phục chiều rộng
            }} else {{
                content.style.display = 'none';
                icon.innerHTML = '+';
                box.style.width = '180px'; // Thu nhỏ chiều rộng
            }}
        }}
    </script>
    """
    return full_html

# --- 5. TẠO ẢNH BACKEND ---
def generate_static_image(df, selected_storms):
    fig = plt.figure(figsize=(14, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([98, 122, 6, 24], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, linestyle="--", edgecolor='gray')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    for storm_name in selected_storms:
        sub = df[df['name'] == storm_name].sort_values('dt')
        if sub.empty: continue
        
        past = sub[sub['status'] != 'forecast']
        forecast = sub[sub['status'] == 'forecast']
        
        if not past.empty:
            ax.plot(past['lon'], past['lat'], transform=ccrs.PlateCarree(), color='blue', linewidth=2, label=f"{storm_name}", zorder=5)
            if not forecast.empty:
                ax.plot([past.iloc[-1]['lon'], forecast.iloc[0]['lon']], [past.iloc[-1]['lat'], forecast.iloc[0]['lat']], transform=ccrs.PlateCarree(), color='red', linewidth=2, linestyle='--', zorder=5)
        if not forecast.empty:
            ax.plot(forecast['lon'], forecast['lat'], transform=ccrs.PlateCarree(), color='red', linewidth=2, linestyle='--', zorder=5)
            
        for _, row in sub.iterrows():
            c = get_color_by_wind(row.get('wind_kt', 0))
            ax.scatter(row['lon'], row['lat'], c=c, s=30, transform=ccrs.PlateCarree(), edgecolor='black', linewidth=0.5, zorder=6)
            
        if not sub.empty:
            start = sub.iloc[0]
            ax.text(start['lon'], start['lat'], storm_name, transform=ccrs.PlateCarree(), fontsize=9, weight='bold', color='darkblue', path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.legend(loc='upper right')
    ax.set_title("SƠ ĐỒ QUỸ ĐẠO BÃO", fontsize=15, weight='bold')
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', dpi=200); buf.seek(0); plt.close(fig)
    return buf

# --- 6. MAIN APP ---

def main():
    with st.sidebar:
        st.title("🌪️ CONTROL PANEL")
        st.markdown("---")
        
        default_file = "besttrack_capgio.xlsx"
        uploaded_file = st.file_uploader("Tải dữ liệu (.xlsx)", type=["xlsx"])
        data_source = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)
        
        if not data_source: st.warning("⚠️ Vui lòng tải file!"); st.stop()
        df = load_data(data_source)
        if df is None or df.empty: st.error("❌ Lỗi dữ liệu."); st.stop()
            
        # FILTER
        st.subheader("🛠️ Bộ lọc")
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            sel_years = st.multiselect("Năm:", years, default=years[-1:])
            temp_df = df[df['year'].isin(sel_years)]
        else: temp_df = df
            
        sel_storms = st.multiselect("Bão:", temp_df['name'].unique(), default=temp_df['name'].unique())
        
        if not temp_df.empty and 'wind_kt' in temp_df.columns:
            min_w, max_w = int(temp_df['wind_kt'].min()), int(temp_df['wind_kt'].max())
            w_range = st.slider("Gió (kt):", min_w, max_w, (min_w, max_w))
            final_df = temp_df[(temp_df['name'].isin(sel_storms)) & (temp_df['wind_kt'] >= w_range[0]) & (temp_df['wind_kt'] <= w_range[1])]
        else: final_df = temp_df
        
        st.markdown("---")
        if not final_df.empty:
            st.download_button("📄 Excel", final_df.to_csv(index=False).encode('utf-8'), "storm_data.csv", "text/csv")
            if st.button("🖼️ Tải ảnh PNG"):
                st.download_button("⬇️ Download PNG", generate_static_image(final_df, sel_storms), "map.png", "image/png")

    # --- MAP SETUP ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None) # Set tiles=None để dùng LayerControl quản lý nền
    
    # 1. Các lớp nền (Base Layers)
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    # 2. Các lớp dữ liệu (Overlays) - Tách nhóm để bật tắt
    fg_past = folium.FeatureGroup(name="📉 Đường đi Thực tế")
    fg_forecast = folium.FeatureGroup(name="🔮 Đường đi Dự báo")
    fg_points = folium.FeatureGroup(name="📍 Điểm và Nhãn")
    
    if not final_df.empty:
        for storm_name in sel_storms:
            storm_data = final_df[final_df['name'] == storm_name].sort_values('dt')
            if storm_data.empty: continue
            
            past_data = storm_data[storm_data['status'] != 'forecast']
            forecast_data = storm_data[storm_data['status'] == 'forecast']
            
            # Vẽ đường Thực tế
            if not past_data.empty:
                coords = past_data[['lat', 'lon']].values.tolist()
                folium.PolyLine(locations=coords, color='black', weight=2, opacity=0.8, tooltip=f"{storm_name} (Past)").add_to(fg_past)
                
            # Vẽ đường Dự báo
            if not forecast_data.empty:
                if not past_data.empty:
                    conn = [[past_data.iloc[-1]['lat'], past_data.iloc[-1]['lon']], [forecast_data.iloc[0]['lat'], forecast_data.iloc[0]['lon']]]
                    folium.PolyLine(locations=conn, color='red', weight=2, dash_array='5, 5').add_to(fg_forecast)
                folium.PolyLine(locations=forecast_data[['lat', 'lon']].values.tolist(), color='red', weight=2, dash_array='5, 5', tooltip=f"{storm_name} (Fcst)").add_to(fg_forecast)

            # Vẽ điểm (Vào nhóm Points để tắt đi cho đỡ rối nếu muốn)
            for _, row in storm_data.iterrows():
                color = get_color_by_wind(row.get('wind_kt', 0))
                popup = f"<b>{row['name']}</b><br>{row['dt'].strftime('%d/%m %Hh')}<br>{int(row.get('wind_kt',0))}kt"
                
                # Nếu là dự báo -> Vào nhóm forecast luôn cho đồng bộ
                target_group = fg_forecast if row['status'] == 'forecast' else fg_points
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']], radius=5 if row['status'] != 'current' else 9,
                    color=color, fill=True, fill_color=color, fill_opacity=1, popup=popup
                ).add_to(target_group)
                
                if row['status'] == 'current': # Hiệu ứng điểm hiện tại
                    folium.CircleMarker([row['lat'], row['lon']], radius=14, color='red', fill=False).add_to(fg_points)

    # Thêm các lớp vào bản đồ
    fg_past.add_to(m)
    fg_forecast.add_to(m)
    fg_points.add_to(m)
    
    # 3. Thêm trình điều khiển Layer (Góc trên phải mặc định)
    folium.LayerControl(collapsed=True).add_to(m)
    
    # Grid vẽ tay (cho vào 1 nhóm riêng để bật tắt)
    fg_grid = folium.FeatureGroup(name="🌐 Lưới Kinh/Vĩ tuyến", show=False)
    for lon in range(100, 131, 5): folium.PolyLine([[0, lon], [35, lon]], color='gray', weight=0.5, dash_array='5').add_to(fg_grid)
    for lat in range(0, 36, 5): folium.PolyLine([[lat, 90], [lat, 140]], color='gray', weight=0.5, dash_array='5').add_to(fg_grid)
    fg_grid.add_to(m)

    # 4. Inject Dashboard HTML
    if not final_df.empty:
        st.markdown(create_dashboard_html(final_df, sel_storms), unsafe_allow_html=True)

    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
