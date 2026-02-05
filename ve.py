import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
from math import radians, sin, cos, asin, sqrt

# --- CẤU HÌNH MÀU SẮC THEO MÃ CỦA PHONG ---
COL_R6   = "#FFC0CB"  # Hồng (Gió cấp 6)
COL_R10  = "#FF6347"  # Đỏ cam (Gió cấp 10)
COL_RC   = "#90EE90"  # Xanh lá (Tâm bão)
COL_TRACK = "black"

st.set_page_config(page_title="Hệ thống Dự báo Bão - Le Van Phong", layout="wide")
st.title("🌀 Bản đồ Hành lang Gió Bão (Hiệu ứng Trong suốt)")

# --- HÀM TÍNH KHOẢNG CÁCH (Haversine) ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# --- HÀM NỘI SUY (10km/bước để tạo dải mịn) ---
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
                'rc': p1.get('bán kính tâm (km)', 0) * (1-frac) + p2.get('bán kính tâm (km)', 0) * frac
            })
    last = df.iloc[-1]
    new_rows.append({'lat': last['lat'], 'lon': last['lon'], 'r6': last.get('bán kính gió mạnh cấp 6 (km)', 0), 
                     'r10': last.get('bán kính gió mạnh cấp 10 (km)', 0), 'rc': last.get('bán kính tâm (km)', 0)})
    return pd.DataFrame(new_rows)

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists("besttrack.xlsx"):
    raw_df = pd.read_excel("besttrack.xlsx")
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])
    dense_df = densify_storm_data(raw_df)

    # Chọn bản đồ nền có độ tương phản tốt để nhìn xuyên thấu
    m = folium.Map(location=[15.8, 112.0], zoom_start=5, tiles="OpenStreetMap")

    # --- VẼ CÁC LỚP TRONG SUỐT (Thứ tự: Ngoài vào Trong) ---
    # Độ trong suốt (fill_opacity) đặt ở mức 0.4 để nhìn được bản đồ bên dưới.

    # 1. Vẽ dải Gió cấp 6 (Dưới cùng)
    for _, row in dense_df.iterrows():
        if row['r6'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['r6']*1000, 
                          color=COL_R6, fill=True, fill_color=COL_R6, 
                          weight=0, fill_opacity=0.4).add_to(m)

    # 2. Vẽ dải Gió cấp 10 (Chồng lên)
    for _, row in dense_df.iterrows():
        if row['r10'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['r10']*1000, 
                          color=COL_R10, fill=True, fill_color=COL_R10,
                          weight=0, fill_opacity=0.4).add_to(m)

    # 3. Vẽ dải Tâm bão (Trên cùng)
    for _, row in dense_df.iterrows():
        if row['rc'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['rc']*1000, 
                          color=COL_RC, fill=True, fill_color=COL_RC,
                          weight=0, fill_opacity=0.5).add_to(m)

    # Đường quỹ đạo chính và Marker
    points = raw_df[['lat', 'lon']].values.tolist()
    if len(points) > 1:
        folium.PolyLine(points, color=COL_TRACK, weight=2, opacity=0.8).add_to(m)

    for _, row in raw_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']], radius=4, color="black", fill=True, fill_opacity=0.9,
            popup=f"Cấp: {row.get('cường độ (cấp BF)', 'N/A')}"
        ).add_to(m)

    st_folium(m, width="100%", height=600)
else:
    st.error("Thiếu file besttrack.xlsx")
