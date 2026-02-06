# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
from math import radians, sin, cos, asin, sqrt
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

# --- CẤU HÌNH HỆ THỐNG ---
ICON_DIR = "icon"
DATA_FILE = "besttrack.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG") 
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90" 

st.set_page_config(page_title="Hệ thống Theo dõi Bão", layout="wide")

# CSS: Cho phép hiện Sidebar nhưng vẫn tràn viền bản đồ
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; height: 100vh !important; }
    header, footer, #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- [HÀM HỖ TRỢ GIỮ NGUYÊN NHƯ CŨ] ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2); dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
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

def get_storm_icon(row):
    status = "daqua" if "quá khứ" in str(row.get('Thời điểm', '')).lower() else "dubao"
    bf = row.get('cường độ (cấp BF)', 0)
    if pd.isna(bf) or bf < 6: fname = f"vungthap{status}.png"
    elif bf < 8: fname = "atnddaqua.PNG" if status == "daqua" else "atnd.PNG"
    elif bf <= 11: fname = "bnddaqua.PNG" if status == "daqua" else "bnd.PNG"
    else: fname = "sieubaodaqua.PNG" if status == "daqua" else "sieubao.PNG"
    path = os.path.join(ICON_DIR, fname)
    return folium.CustomIcon(path, icon_size=(35, 35) if bf >= 8 else (22, 22)) if os.path.exists(path) else None

def create_storm_swaths(dense_df):
    polys_r6, polys_r10, polys_rc = [], [], []
    geo = geodesic.Geodesic()
    for _, row in dense_df.iterrows():
        for r, target_list in [(row.get('r6', 0), polys_r6), (row.get('r10', 0), polys_r10), (row.get('rc', 0), polys_rc)]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=60)
                target_list.append(Polygon(circle))
    u6 = unary_union(polys_r6) if polys_r6 else None
    u10 = unary_union(polys_r10) if polys_r10 else None
    uc = unary_union(polys_rc) if polys_rc else None
    f_rc = uc
    f_r10 = u10.difference(uc) if u10 and uc else u10
    f_r6 = u6.difference(u10) if u6 and u10 else u6
    return f_r6, f_r10, f_rc

# --- 2. LOGIC LỌC BÃO ---
if os.path.exists(DATA_FILE):
    full_df = pd.read_excel(DATA_FILE)
    full_df[['lat', 'lon']] = full_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    full_df = full_df.dropna(subset=['lat', 'lon'])

    # Giả sử file có cột 'Số hiệu' hoặc 'Tên bão'. Nếu không có, ta coi như 1 cơn bão.
    storm_col = 'Số hiệu' if 'Số hiệu' in full_df.columns else None
    
    if storm_col:
        st.sidebar.title("Danh sách bão")
        storm_list = full_df[storm_col].unique()
        selected_storms = [s for s in storm_list if st.sidebar.checkbox(f"Bão {s}", value=True)]
        filtered_df = full_df[full_df[storm_col].isin(selected_storms)]
    else:
        filtered_df = full_df
        st.sidebar.info("Không tìm thấy cột 'Số hiệu' để phân loại bão.")

    # --- 3. HIỂN THỊ BẢN ĐỒ ---
    m = folium.Map(location=[17.0, 115.0], zoom_start=5, tiles="OpenStreetMap")
    
    # Vẽ từng cơn bão đã chọn
    if storm_col and not filtered_df.empty:
        for s_id in selected_storms:
            storm_data = filtered_df[filtered_df[storm_col] == s_id]
            dense_df = densify_track(storm_data)
            
            # Layer cho từng cơn bão
            fg_storm = folium.FeatureGroup(name=f"Bão {s_id}")
            
            # Quét vùng gió
            f6, f10, fc = create_storm_swaths(dense_df)
            for geom, col, op in [(f6, COL_R6, 0.4), (f10, COL_R10, 0.5), (fc, COL_RC, 0.6)]:
                if geom and not geom.is_empty:
                    folium.GeoJson(mapping(geom), style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg_storm)
            
            # Đường đi
            folium.PolyLine(storm_data[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(fg_storm)
            
            # Icon
            for _, row in storm_data.iterrows():
                icon = get_storm_icon(row)
                if icon: folium.Marker([row['lat'], row['lon']], icon=icon).add_to(fg_storm)
            
            fg_storm.add_to(m)
    
    # Layer Control & Legend
    folium.LayerControl(position='topleft').add_to(m)
    
    # Giao diện dashboard bên phải (chỉ hiện dữ liệu của bão cuối cùng được chọn hoặc bão chính)
    if not filtered_df.empty:
        # (Hàm get_right_dashboard_html giữ nguyên như code trước của bạn)
        pass 

    st_folium(m, width=None, height=1000, use_container_width=True)
else:
    st.error("Thiếu file besttrack.xlsx")
