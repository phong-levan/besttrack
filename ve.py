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

# Thư viện vẽ đồ thị tĩnh (Export ảnh)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patheffects as path_effects

# Thư viện xử lý hình học
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & GIAO DIỆN
# ==============================================================================
st.set_page_config(
    page_title="Hệ thống Tích hợp Bão & Thời tiết",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Full Screen & Giao diện
st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: transparent !important; }
    header, footer { display: none !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; z-index: 0; }
    
    /* Sidebar nổi lên trên */
    [data-testid="stSidebar"] { z-index: 10000 !important; background-color: rgba(28, 35, 49, 0.95) !important; }
    
    /* Style cho các hộp thông tin */
    .info-box { z-index: 9999 !important; }
    </style>
""", unsafe_allow_html=True)

# Cấu hình file
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"        # Bão Hiện trạng
FILE_OPT2 = "besttrack_capgio.xlsx" # Bão Lịch sử
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU BÃO (GIỮ NGUYÊN TỪ PHẦN TRƯỚC)
# ==============================================================================

@st.cache_data
def load_storm_data(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_excel(file_path)
    
    # Rename columns standard
    rename_map = {
        "tên bão": "name", "biển đông": "storm_no", "năm": "year", "tháng": "mon", 
        "ngày": "day", "giờ": "hour", "vĩ độ": "lat", "kinh độ": "lon", 
        "gió (kt)": "wind_kt", "khí áp (mb)": "pressure", 
        "Thời điểm": "status_raw", "Ngày - giờ": "datetime_str",
        "bán kính gió mạnh cấp 6 (km)": "r6", "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    # Process DateTime
    if 'datetime_str' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    elif all(c in df.columns for c in ['year', 'mon', 'day', 'hour']):
        cols = ['year', 'mon', 'day', 'hour']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=cols)
        df[cols] = df[cols].astype(int)
        df['dt'] = pd.to_datetime(df[cols].rename(columns={'mon':'month'}))
        
    num_cols = ['lat', 'lon', 'wind_kt', 'r6', 'r10', 'rc']
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df.dropna(subset=['lat', 'lon'])

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

def generate_static_map(df, title="SƠ ĐỒ BÃO"):
    fig = plt.figure(figsize=(14, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([98, 125, 5, 25], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, linestyle="--", edgecolor='gray')
    
    unique_storms = df['name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_storms)))
    
    for i, storm_name in enumerate(unique_storms):
        sub = df[df['name'] == storm_name].sort_values('dt')
        if sub.empty: continue
        color = colors[i]
        ax.plot(sub['lon'], sub['lat'], color=color, linewidth=2, transform=ccrs.PlateCarree(), label=storm_name, zorder=5)
        ax.scatter(sub['lon'], sub['lat'], color=color, s=20, zorder=6, transform=ccrs.PlateCarree())
        start_pt = sub.iloc[0]
        ax.text(start_pt['lon'], start_pt['lat'], storm_name, transform=ccrs.PlateCarree(), fontsize=9, color=color, weight='bold', path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.legend(loc='upper right', title="Danh sách bão")
    ax.set_title(title, fontsize=15, weight='bold', color='#003366', pad=15)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0); plt.close(fig)
    return buf

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH (LOGIC PHÂN CẤP)
# ==============================================================================
def main():
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None)
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    fg_storm = folium.FeatureGroup(name="Lớp Bão")
    fg_weather = folium.FeatureGroup(name="Lớp Thời Tiết")

    with st.sidebar:
        st.title("🎛️ MENU ĐIỀU KHIỂN")
        
        # --- CẤP 1: CHỌN CHỦ ĐỀ (BÃO vs THỜI TIẾT) ---
        main_topic = st.selectbox("1. CHỌN CHỦ ĐỀ:", ["Bão (Typhoon)", "Thời tiết (Weather)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        
        # ==================== NHÁNH 1: BÃO ====================
        if main_topic == "Bão (Typhoon)":
            storm_mode = st.radio("2. CHỨC NĂNG BÃO:", ["Option 1: Hiện trạng", "Option 2: Lịch sử"])
            
            # --- Option 1: Hiện trạng ---
            if "Option 1" in storm_mode:
                st.info("📂 Đang dùng: besttrack.xlsx")
                f = st.file_uploader("Upload File:", type="xlsx", key="opt1")
                path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                
                show_layer = st.checkbox("Hiển thị lớp Hiện trạng", value=True)
                
                if path:
                    df = load_storm_data(path)
                    if df is not None and not df.empty:
                        # Logic Option 1
                        if 'storm_no' in df.columns:
                            selected = st.multiselect("Chọn bão:", df['storm_no'].unique(), default=df['storm_no'].unique())
                            final_df = df[df['storm_no'].isin(selected)]
                        else: final_df = df
                        
                        if show_layer and not final_df.empty:
                            groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
                            for g in groups:
                                sub = final_df[final_df['storm_no'] == g] if g else final_df
                                if sub.empty: continue
                                dense = densify_track(sub)
                                f6, f10, fc = create_storm_swaths(dense)
                                for geom, c, o in [(f6,COL_R6,0.4), (f10,COL_R10,0.5), (fc,COL_RC,0.6)]:
                                    if geom and not geom.is_empty:
                                        folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':0,'fillOpacity':o}).add_to(fg_storm)
                                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_storm)
                else:
                    st.warning("Vui lòng tải file besttrack.xlsx")

            # --- Option 2: Lịch sử ---
            else:
                st.info("📂 Đang dùng: besttrack_capgio.xlsx")
                f = st.file_uploader("Upload File:", type="xlsx", key="opt2")
                path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
                
                show_layer = st.checkbox("Hiển thị lớp Lịch sử", value=True)
                
                if path:
                    df = load_storm_data(path)
                    if df is not None and not df.empty:
                        # Logic Option 2 Filters
                        years = st.multiselect("Lọc Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                        temp = df[df['year'].isin(years)]
                        names = st.multiselect("Lọc Tên Bão:", temp['name'].unique(), default=temp['name'].unique())
                        final_df = temp[temp['name'].isin(names)]
                        
                        if show_layer and not final_df.empty:
                            for name in final_df['name'].unique():
                                sub = final_df[final_df['name'] == name].sort_values('dt')
                                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2, opacity=0.6).add_to(fg_storm)
                                for _, r in sub.iterrows():
                                    folium.CircleMarker([r['lat'],r['lon']], radius=3, color='red', fill=True, popup=f"{name}").add_to(fg_storm)
                else:
                    st.warning("Vui lòng tải file besttrack_capgio.xlsx")
            
            # Export cho Bão
            if not final_df.empty:
                st.markdown("---")
                if st.button("🖼️ Tải ảnh bản đồ Bão (PNG)"):
                    img = generate_static_map(final_df, title=f"BẢN ĐỒ BÃO - {storm_mode}")
                    st.download_button("⬇️ Download PNG", img, "storm_map.png", "image/png")

        # ==================== NHÁNH 2: THỜI TIẾT ====================
        else:
            weather_mode = st.radio("2. LOẠI DỮ LIỆU:", ["Option 3: Quan trắc", "Option 4: Mô hình"])
            
            # Sub-options chung cho cả Quan trắc và Mô hình
            param = st.selectbox("3. CHỌN THÔNG SỐ:", [
                "Nhiệt độ (Option 5/8)", 
                "Lượng mưa (Option 6/9)", 
                "Gió (Option 7/10)"
            ])
            
            st.markdown("---")
            show_weather = st.checkbox(f"Hiển thị lớp {param}", value=True)
            
            # --- Logic Giả lập cho Thời tiết (Vì chưa có file dữ liệu thật) ---
            if show_weather:
                # Đây là demo hạ tầng: Hiển thị các lớp giả lập để chứng minh luồng hoạt động
                st.success(f"Đang hiển thị lớp: {weather_mode} > {param}")
                
                if "Nhiệt độ" in param:
                    # Demo: Heatmap
                    from folium.plugins import HeatMap
                    data = [[16, 108, 25], [10, 106, 30], [21, 105, 20]] # Fake data
                    HeatMap(data, radius=15).add_to(fg_weather)
                    st.caption("ℹ️ (Demo Heatmap Nhiệt độ)")
                    
                elif "Lượng mưa" in param:
                    # Demo: Marker Cluster
                    for lat, lon in [(16, 108), (12, 109), (20, 106)]:
                        folium.Circle([lat, lon], radius=20000, color='blue', fill=True, fill_opacity=0.3).add_to(fg_weather)
                    st.caption("ℹ️ (Demo Vùng mưa)")
                    
                elif "Gió" in param:
                    # Demo: Vector lines
                    for lat in range(10, 20, 2):
                        folium.PolyLine([[lat, 110], [lat+0.5, 111]], color='green', weight=2, opacity=0.8).add_to(fg_weather)
                    st.caption("ℹ️ (Demo Hướng gió)")

    # --- RENDER MAP ---
    fg_storm.add_to(m)
    fg_weather.add_to(m)
    folium.LayerControl(position='bottomleft', collapsed=True).add_to(m)
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
