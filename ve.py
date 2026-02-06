# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

# Cố gắng import các plugin, nếu lỗi sẽ tạm ẩn tính năng đó để tránh sập app
try:
    from folium.plugins import SimpleScreenshot, MousePosition, MeasureControl
    HAS_PLUGINS = True
except ImportError:
    HAS_PLUGINS = False

# --- 1. CẤU HÌNH HỆ THỐNG (Giữ nguyên từ file làm web.docx) ---
DATA_FOLDER = "besttrack"  
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(page_title="Hệ thống Theo dõi Bão", layout="wide")

# CSS: GIỮ NGUYÊN CODE NỀN (Full màn hình, tràn viền)
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        overflow: hidden !important; height: 100vh !important; width: 100vw !important; margin: 0 !important; padding: 0 !important;
    }
    .main .block-container { padding: 0 !important; max-width: 100% !important; height: 100vh !important; }
    [data-testid="stHeader"], footer { display: none !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; border: none !important; z-index: 1; }
    [data-testid="stSidebar"] { z-index: 100; background-color: rgba(248, 249, 250, 0.95); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM HỖ TRỢ KỸ THUẬT ---
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
        n_steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(n_steps):
            f = j / n_steps
            new_rows.append({
                'lat': p1['lat'] + (p2['lat'] - p1['lat']) * f,
                'lon': p1['lon'] + (p2['lon'] - p1['lon']) * f,
                'r6': p1.get('bán kính gió mạnh cấp 6 (km)', 0)*(1-f) + p2.get('bán kính gió mạnh cấp 6 (km)', 0)*f,
                'r10': p1.get('bán kính gió mạnh cấp 10 (km)', 0)*(1-f) + p2.get('bán kính gió mạnh cấp 10 (km)', 0)*f,
                'rc': p1.get('bán kính tâm (km)', 0)*(1-f) + p2.get('bán kính tâm (km)', 0)*f
            })
    new_rows.append(df.iloc[-1].to_dict())
    return pd.DataFrame(new_rows)

# --- 3. MODULE CODE NỀN (BASE MAP) ---
def create_base_map():
    m = folium.Map(location=[17.5, 115.0], zoom_start=6, tiles="OpenStreetMap")
    
    # Vẽ lưới kinh vĩ độ
    for lon in range(100, 141, 5):
        folium.PolyLine([[0, lon], [40, lon]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    for lat in range(0, 41, 5):
        folium.PolyLine([[lat, 100], [lat, 140]], color='gray', weight=0.5, opacity=0.3).add_to(m)
    
    # Chỉ thêm plugin nếu import thành công
    if HAS_PLUGINS:
        MousePosition().add_to(m)
        MeasureControl(primary_length_unit='kilometers').add_to(m)
        SimpleScreenshot().add_to(m)
    else:
        st.sidebar.warning("⚠️ Đang tải thư viện bổ sung, một số công cụ (thước đo, tọa độ) tạm ẩn.")
    
    return m

# --- 4. MODULE CON (VẼ DỮ LIỆU) ---
def draw_storm_data(fg, df_raw, color="black", show_swaths=False):
    if show_swaths:
        df_dense = densify_track(df_raw)
        polys_r6, polys_r10, polys_rc = [], [], []
        geo = geodesic.Geodesic()
        for _, row in df_dense.iterrows():
            for r, target in [(row.get('r6', 0), polys_r6), (row.get('r10', 0), polys_r10), (row.get('rc', 0), polys_rc)]:
                if r > 0:
                    circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                    target.append(Polygon(circle))
        u6, u10, uc = unary_union(polys_r6), unary_union(polys_r10), unary_union(polys_rc)
        for geom, col, op in [(u6, COL_R6, 0.4), (u10, COL_R10, 0.5), (uc, COL_RC, 0.6)]:
            if geom and not geom.is_empty:
                folium.GeoJson(mapping(geom), style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg)
    
    points = df_raw[['lat', 'lon']].values.tolist()
    folium.PolyLine(points, color=color, weight=2, opacity=0.8).add_to(fg)

# --- 5. LOGIC ĐIỀU KHIỂN (OPTION 1 & 2) ---
m = create_base_map()
st.sidebar.title("🎛️ Tùy chọn")

if os.path.exists(DATA_FOLDER):
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    # Option 1: Bão hiện tại
    st.sidebar.subheader("📍 Option 1")
    sel_curr = st.sidebar.multiselect("Chọn bão hiện trạng:", options=all_files, default=all_files[:1] if all_files else [])
    
    # Option 2: Lọc quá khứ
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕰️ Option 2")
    
    all_dfs = []
    for f in all_files:
        try:
            all_dfs.append(pd.read_excel(os.path.join(DATA_FOLDER, f), sheet_name='besttrack'))
        except: pass
    
    if all_dfs:
        combined = pd.concat(all_dfs).dropna(subset=['lat', 'lon'])
        combined[['lat', 'lon']] = combined[['lat', 'lon']].apply(pd.to_numeric)
        
        sel_nums = st.sidebar.multiselect("Lọc bão quá khứ:", options=sorted(combined['Số hiệu'].unique().tolist()))
        
        # Vẽ Option 1
        for f_name in sel_curr:
            df_tmp = pd.read_excel(os.path.join(DATA_FOLDER, f_name), sheet_name='besttrack')
            df_curr = df_tmp[df_tmp['Thời điểm'].str.contains("hiện tại|dự báo", case=False, na=False)]
            if not df_curr.empty:
                fg = folium.FeatureGroup(name=f"Hiện tại: {f_name}")
                draw_storm_data(fg, df_curr, color="red", show_swaths=True)
                fg.add_to(m)

        # Vẽ Option 2
        if sel_nums:
            df_filt = combined[combined['Số hiệu'].isin(sel_nums)]
            if not df_filt.empty:
                fg_p = folium.FeatureGroup(name="Dữ liệu lọc")
                draw_storm_data(fg_p, df_filt, color="blue", show_swaths=False)
                fg_p.add_to(m)

folium.LayerControl(position='topleft').add_to(m)
st_folium(m, width=2500, height=1200, use_container_width=True)
