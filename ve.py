import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
from math import radians, sin, cos, asin, sqrt

# --- CẤU HÌNH MÀU SẮC (Theo ảnh tĩnh của Phong) ---
COL_R6   = "#FFC0CB"  # Hồng (Gió cấp 6) - Lớp dưới cùng
COL_R10  = "#FF6347"  # Đỏ cam (Gió cấp 10) - Lớp giữa
COL_RC   = "#90EE90"  # Xanh lá (Tâm bão) - Lớp trên cùng
COL_TRACK = "black"

st.set_page_config(page_title="Hệ thống Nội suy Bão - Phong Le", layout="wide")
st.title("🌀 Bản đồ Hành lang Gió Bão (Xếp lớp chuẩn)")

# --- HÀM TÍNH KHOẢNG CÁCH HAVERSINE ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# --- HÀM NỘI SUY DỌC ĐƯỜNG ĐI (Bước 10km để tạo dải liền mạch) ---
def densify_storm_data(df, step_km=10):
    new_rows = []
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine_km(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        n_steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(n_steps):
            frac = j / n_steps
            new_rows.append({
                'lat': p1['lat'] + (p2['lat'] - p1['lat']) * frac,
                'lon': p1['lon'] + (p2['lon'] - p1['lon']) * frac,
                'r6': p1.get('bán kính gió mạnh cấp 6 (km)', 0) * (1-frac) + p2.get('bán kính gió mạnh cấp 6 (km)', 0) * frac,
                'r10': p1.get('bán kính gió mạnh cấp 10 (km)', 0) * (1-frac) + p2.get('bán kính gió mạnh cấp 10 (km)', 0) * frac,
                'rc': p1.get('bán kính tâm (km)', 0) * (1-frac) + p2.get('bán kính tâm (km)', 0) * frac,
                'is_original': j == 0 # Đánh dấu điểm gốc
            })
    # Thêm điểm cuối cùng
    last = df.iloc[-1]
    new_rows.append({
        'lat': last['lat'], 'lon': last['lon'],
        'r6': last.get('bán kính gió mạnh cấp 6 (km)', 0),
        'r10': last.get('bán kính gió mạnh cấp 10 (km)', 0),
        'rc': last.get('bán kính tâm (km)', 0),
        'is_original': True
    })
    return pd.DataFrame(new_rows)

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists("besttrack.xlsx"):
    raw_df = pd.read_excel("besttrack.xlsx")
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])
    
    # 1. Thực hiện nội suy
    dense_df = densify_storm_data(raw_df)

    m = folium.Map(location=[15.8, 112.0], zoom_start=5, tiles="CartoDB positron")

    # --- VẼ CÁC LỚP MÀU THEO THỨ TỰ (QUAN TRỌNG) ---
    # Chúng ta dùng fill_opacity=1.0 để lớp trên che hoàn toàn lớp dưới.

    # Lớp 1: Vẽ tất cả vòng tròn Cấp 6 (Hồng) trước - Nằm dưới cùng
    for _, row in dense_df.iterrows():
        if row['r6'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['r6']*1000, 
                          color=COL_R6, fill=True, fill_color=COL_R6, 
                          weight=0, fill_opacity=1.0).add_to(m)

    # Lớp 2: Vẽ tất cả vòng tròn Cấp 10 (Đỏ) chồng lên
    for _, row in dense_df.iterrows():
        if row['r10'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['r10']*1000, 
                          color=COL_R10, fill=True, fill_color=COL_R10,
                          weight=0, fill_opacity=1.0).add_to(m)

    # Lớp 3: Vẽ tất cả vòng tròn Tâm bão (Xanh) chồng lên trên cùng
    for _, row in dense_df.iterrows():
        if row['rc'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['rc']*1000, 
                          color=COL_RC, fill=True, fill_color=COL_RC,
                          weight=0, fill_opacity=1.0).add_to(m)

    # --- Vẽ đường đi và các điểm mốc ---
    # Vẽ đường nối tâm bão
    points = raw_df[['lat', 'lon']].values.tolist()
    if len(points) > 1:
        folium.PolyLine(points, color=COL_TRACK, weight=2.5, opacity=1.0, z_index=1000).add_to(m)

    # Vẽ các Marker tại điểm gốc (để click xem thông tin)
    for _, row in raw_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']], radius=5, color="black", fill=True, fill_opacity=1.0,
            popup=f"Thời gian: {row.get('Ngày - giờ', 'N/A')}<br>Cấp: {row.get('cường độ (cấp BF)', 'N/A')}",
            z_index=1001 # Đảm bảo marker luôn nổi lên trên cùng
        ).add_to(m)

    st_folium(m, width="100%", height=600)
else:
    st.error("Vui lòng kiểm tra file besttrack.xlsx")
