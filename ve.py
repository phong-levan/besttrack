# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import io
import warnings
from math import radians, sin, cos, asin, sqrt

# Thư viện vẽ đồ thị tĩnh (Export ảnh chất lượng cao)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patheffects as path_effects

# Thư viện xử lý hình học (Vùng gió)
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & GIAO DIỆN CƠ BẢN (SOURCE 2)
# ==============================================================================
st.set_page_config(
    page_title="Hệ thống Giám sát Bão Biển Đông",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Full Screen
st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: transparent !important; }
    header, footer { display: none !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; z-index: 0; }
    [data-testid="stSidebar"] { z-index: 10000 !important; background-color: rgba(28, 35, 49, 0.95) !important; }
    .dashboard-box { z-index: 9999 !important; }
    </style>
""", unsafe_allow_html=True)

# Cấu hình file & màu sắc
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"
FILE_OPT2 = "besttrack_capgio.xlsx"
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU & TÍNH TOÁN (SOURCE 4 & 8)
# ==============================================================================

@st.cache_data
def load_data(file_path, mode="opt2"):
    if not os.path.exists(file_path): return None
    
    if file_path.endswith('.csv'): df = pd.read_csv(file_path)
    else: df = pd.read_excel(file_path)
    
    # Chuẩn hóa tên cột
    rename_map = {
        "tên bão": "name", "biển đông": "storm_no", "năm": "year", "tháng": "mon", 
        "ngày": "day", "giờ": "hour", "vĩ độ": "lat", "kinh độ": "lon", 
        "gió (kt)": "wind_kt", "khí áp (mb)": "pressure", 
        "Thời điểm": "status_raw", "Ngày - giờ": "datetime_str",
        "bán kính gió mạnh cấp 6 (km)": "r6", "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    # Xử lý thời gian
    if 'datetime_str' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    elif all(c in df.columns for c in ['year', 'mon', 'day', 'hour']):
        cols = ['year', 'mon', 'day', 'hour']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=cols)
        df[cols] = df[cols].astype(int)
        df['dt'] = pd.to_datetime(df[cols].rename(columns={'mon':'month'}))
        
    # Ép kiểu số
    num_cols = ['lat', 'lon', 'wind_kt', 'pressure', 'r6', 'r10', 'rc']
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df.dropna(subset=['lat', 'lon'])

# --- Hàm tính toán vùng gió (Option 1) ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=10):
    new_rows = []
    if len(df) < 2: return df
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine_km(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(steps):
            f = j / steps
            row = p1.copy()
            row['lat'] = p1['lat'] + (p2['lat'] - p1['lat']) * f
            row['lon'] = p1['lon'] + (p2['lon'] - p1['lon']) * f
            for col in ['r6', 'r10', 'rc']:
                if col in p1 and col in p2: row[col] = p1.get(col, 0)*(1-f) + p2.get(col, 0)*f
            new_rows.append(row)
    new_rows.append(df.iloc[-1])
    return pd.DataFrame(new_rows)

def create_storm_swaths(dense_df):
    polys = {'r6': [], 'r10': [], 'rc': []}
    geo = geodesic.Geodesic()
    for _, row in dense_df.iterrows():
        for r, key in [(row.get('r6',0), 'r6'), (row.get('r10',0), 'r10'), (row.get('rc',0), 'rc')]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                polys[key].append(Polygon(circle))
    u = {k: unary_union(v) if v else None for k, v in polys.items()}
    f_rc = u['rc']
    f_r10 = u['r10'].difference(u['rc']) if u['r10'] and u['rc'] else u['r10']
    f_r6 = u['r6'].difference(u['r10']) if u['r6'] and u['r10'] else u['r6']
    return f_r6, f_r10, f_rc

# --- Hàm sinh ảnh bản đồ tĩnh chất lượng cao (Source 8 - Cartopy) ---
def generate_static_map(df, title="SƠ ĐỒ BÃO"):
    fig = plt.figure(figsize=(14, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([98, 125, 5, 25], crs=ccrs.PlateCarree())
    
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
    ax.gridlines(draw_labels=False, linewidth=0.5, color="gray", alpha=0.3, linestyle="--")
    
    # Vẽ bão
    unique_storms = df['name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_storms)))
    
    for i, storm_name in enumerate(unique_storms):
        sub = df[df['name'] == storm_name].sort_values('dt')
        if sub.empty: continue
        
        color = colors[i]
        # Đường đi
        ax.plot(sub['lon'], sub['lat'], color=color, linewidth=2, transform=ccrs.PlateCarree(), label=storm_name, zorder=5)
        # Điểm
        ax.scatter(sub['lon'], sub['lat'], color=color, s=20, zorder=6, transform=ccrs.PlateCarree())
        # Tên
        start_pt = sub.iloc[0]
        ax.text(start_pt['lon'], start_pt['lat'], storm_name, transform=ccrs.PlateCarree(), 
                fontsize=9, color=color, weight='bold', path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.legend(loc='upper right', title="Danh sách bão")
    ax.set_title(title, fontsize=15, weight='bold', color='#003366', pad=15)
    
    # Lưu vào buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    plt.close(fig)
    return buf

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH (MAIN APP)
# ==============================================================================
def main():
    # --- SIDEBAR CONTROL PANEL (Theo Flowchart) ---
    with st.sidebar:
        st.title("🌪️ CONTROL PANEL")
        
        # ROOT: Chọn chế độ
        option = st.radio("📍 CHỌN CHẾ ĐỘ DỮ LIỆU:", ["Option 1: Hiện trạng & Dự báo", "Option 2: Lịch sử & Thống kê"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        selected_storms = []
        show_layer = True
        
        # --- NHÁNH 1: OPTION 1 (HIỆN TRẠNG) ---
        if "Option 1" in option:
            st.info("📂 Nguồn: besttrack.xlsx")
            uploaded = st.file_uploader("Tải file dữ liệu:", type=["xlsx"], key="opt1")
            file_path = uploaded if uploaded else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
            
            if file_path:
                df = load_data(file_path, mode="opt1")
                
                # Checkbox: Bật/tắt lớp layer hiện trạng (Theo flowchart)
                show_layer = st.checkbox("Hiển thị lớp Hiện trạng/Dự báo", value=True)
                
                if df is not None and not df.empty:
                    # Lọc bão (nếu có cột số hiệu)
                    if 'storm_no' in df.columns:
                        all_s = df['storm_no'].unique()
                        selected_storms = [s for s in all_s if st.checkbox(f"Bão số {s}", value=True)]
                        final_df = df[df['storm_no'].isin(selected_storms)]
                    else:
                        final_df = df # Mặc định lấy hết nếu không có số hiệu
            else:
                st.warning("Vui lòng tải file besttrack.xlsx")

        # --- NHÁNH 2: OPTION 2 (LỊCH SỬ) ---
        else:
            st.info("📂 Nguồn: besttrack_capgio.xlsx")
            uploaded = st.file_uploader("Tải file dữ liệu:", type=["xlsx"], key="opt2")
            file_path = uploaded if uploaded else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
            
            if file_path:
                df = load_data(file_path, mode="opt2")
                
                if df is not None and not df.empty:
                    # Checkbox: Bật/tắt lớp layer quá khứ (Theo flowchart)
                    show_layer = st.checkbox("Hiển thị lớp Lịch sử", value=True)
                    
                    st.markdown("### 🛠️ Bộ lọc dữ liệu")
                    # 1. Lọc Năm
                    years = sorted(df['year'].unique())
                    sel_years = st.multiselect("Chọn Năm:", years, default=years[-1:] if years else None)
                    temp_df = df[df['year'].isin(sel_years)]
                    
                    # 2. Lọc Tháng
                    if not temp_df.empty:
                        months = sorted(temp_df['mon'].unique())
                        sel_months = st.multiselect("Chọn Tháng:", months, default=months)
                        temp_df = temp_df[temp_df['mon'].isin(sel_months)]
                    
                    # 3. Lọc Tên Bão
                    if not temp_df.empty:
                        names = temp_df['name'].unique()
                        sel_names = st.multiselect("Chọn Tên Bão:", names, default=names)
                        temp_df = temp_df[temp_df['name'].isin(sel_names)]
                        selected_storms = sel_names # Để dùng cho logic vẽ sau này
                    
                    # 4. Lọc Cường độ (Slider)
                    if not temp_df.empty:
                        min_w, max_w = int(temp_df['wind_kt'].min()), int(temp_df['wind_kt'].max())
                        w_range = st.slider("Phạm vi gió (kt):", min_w, max_w, (min_w, max_w))
                        final_df = temp_df[(temp_df['wind_kt'] >= w_range[0]) & (temp_df['wind_kt'] <= w_range[1])]
            else:
                st.warning("Vui lòng tải file besttrack_capgio.xlsx")

        # NÚT DOWNLOAD (CHUNG CHO CẢ 2 OPTION)
        st.markdown("---")
        if not final_df.empty:
            st.success(f"Đang hiển thị: {len(final_df)} điểm dữ liệu.")
            # Excel Export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📥 Tải dữ liệu lọc (.xlsx)", output.getvalue(), "data_loc.xlsx")
            
            # Image Export (Matplotlib/Cartopy)
            if st.button("🖼️ Tạo & Tải ảnh bản đồ (PNG)"):
                with st.spinner("Đang vẽ bản đồ chất lượng cao (Cartopy)..."):
                    img_buf = generate_static_map(final_df, title=f"BẢN ĐỒ BÃO ({'Hiện trạng' if 'Option 1' in option else 'Lịch sử'})")
                    st.download_button("⬇️ Tải ảnh PNG", img_buf, "storm_map.png", "image/png")

    # ==============================================================================
    # 4. KHỞI TẠO BẢN ĐỒ WEB (INTERACTIVE MAP)
    # ==============================================================================
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None)
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    fg_data = folium.FeatureGroup(name="Dữ liệu Bão")

    # LOGIC VẼ LÊN BẢN ĐỒ DỰA THEO OPTION
    if not final_df.empty and show_layer:
        
        # --- VẼ OPTION 1: VÙNG GIÓ + ĐƯỜNG ĐI ---
        if "Option 1" in option:
            # Nhóm theo từng cơn bão (dựa trên storm_no hoặc name)
            groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
            
            for g in groups:
                sub = final_df[final_df['storm_no'] == g] if g else final_df
                if sub.empty: continue
                
                # 1. Vẽ vùng gió (Polygons)
                dense = densify_track(sub)
                f6, f10, fc = create_storm_swaths(dense)
                for geom, col, op in [(f6,COL_R6,0.4), (f10,COL_R10,0.5), (fc,COL_RC,0.6)]:
                    if geom and not geom.is_empty:
                        folium.GeoJson(mapping(geom), 
                                     style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':0,'fillOpacity':o}
                                     ).add_to(fg_data)
                
                # 2. Vẽ đường đi
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_data)
                
                # 3. Vẽ Icon/Marker
                for _, row in sub.iterrows():
                    # Logic lấy icon (nếu có file icon) hoặc vẽ chấm
                    # Ở đây dùng chấm tròn đơn giản cho demo, bạn có thể ghép logic get_icon_name cũ vào
                    folium.CircleMarker(
                        [row['lat'], row['lon']], radius=4, color='red', fill=True,
                        popup=f"Bão số {row.get('storm_no','')}: {row.get('wind_kt','')}kt"
                    ).add_to(fg_data)

        # --- VẼ OPTION 2: ĐƯỜNG ĐI LỊCH SỬ ---
        else:
            for storm_name in final_df['name'].unique():
                sub = final_df[final_df['name'] == storm_name].sort_values('dt')
                if sub.empty: continue
                
                # Vẽ đường đi
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2, opacity=0.6).add_to(fg_data)
                
                # Vẽ các điểm
                for _, row in sub.iterrows():
                    # Màu theo gió
                    w = row.get('wind_kt', 0)
                    c = '#00CCFF' if w<34 else ('#00FF00' if w<64 else ('#FFFF00' if w<83 else '#FF0000'))
                    
                    folium.CircleMarker(
                        [row['lat'], row['lon']], radius=4, color=c, fill=True, fill_opacity=1,
                        popup=f"{storm_name}: {int(w)}kt ({row['dt'].strftime('%Y-%m-%d')})"
                    ).add_to(fg_data)

    fg_data.add_to(m)
    
    # Layer Control ở góc dưới trái (để tránh che bảng tin bên phải)
    folium.LayerControl(position='bottomleft', collapsed=True).add_to(m)
    
    # Hiển thị Full Screen
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
