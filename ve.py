import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
from math import radians, sin, cos, asin, sqrt

# --- CẤU HÌNH MÀU SẮC THEO YÊU CẦU CỦA PHONG ---
COL_R6   = "#FFC0CB"  # Hồng
COL_R10  = "#FF6347"  # Đỏ cam
COL_RC   = "#90EE90"  # Xanh lá
COL_TRACK = "black"

st.set_page_config(page_title="Theo dõi xoáy thuận nhiệt đới", layout="wide")
st.title("🌀 Theo dõi xoáy thuận nhiệt đới")

# --- HÀM TÍNH KHOẢNG CÁCH HAVERSINE ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# --- HÀM NỘI SUY DỌC ĐƯỜNG ĐI (DENSIFY TRACK) ---
def densify_storm_data(df, step_km=10):
    new_rows = []
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine_km(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        
        # Tính số điểm nội suy cần thiết
        n_steps = max(1, int(np.ceil(dist / step_km)))
        
        for j in range(n_steps):
            frac = j / n_steps
            interp_row = {
                'lat': p1['lat'] + (p2['lat'] - p1['lat']) * frac,
                'lon': p1['lon'] + (p2['lon'] - p1['lon']) * frac,
                'r6': p1.get('bán kính gió mạnh cấp 6 (km)', 0) * (1-frac) + p2.get('bán kính gió mạnh cấp 6 (km)', 0) * frac,
                'r10': p1.get('bán kính gió mạnh cấp 10 (km)', 0) * (1-frac) + p2.get('bán kính gió mạnh cấp 10 (km)', 0) * frac,
                'rc': p1.get('bán kính tâm (km)', 0) * (1-frac) + p2.get('bán kính tâm (km)', 0) * frac,
                'is_interp': True if j > 0 else False
            }
            new_rows.append(interp_row)
            
    new_rows.append({
        'lat': df.iloc[-1]['lat'], 'lon': df.iloc[-1]['lon'],
        'r6': df.iloc[-1].get('bán kính gió mạnh cấp 6 (km)', 0),
        'r10': df.iloc[-1].get('bán kính gió mạnh cấp 10 (km)', 0),
        'rc': df.iloc[-1].get('bán kính tâm (km)', 0),
        'is_interp': False
    })
    return pd.DataFrame(new_rows)

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists("besttrack.xlsx"):
    raw_df = pd.read_excel("besttrack.xlsx")
    # Xử lý dữ liệu trống
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])
    
    # Thực hiện nội suy dày đặc
    dense_df = densify_storm_data(raw_df)

    m = folium.Map(location=[15.8, 112.0], zoom_start=5, tiles="CartoDB positron")

    # Vẽ hành lang gió từ dữ liệu nội suy
    for _, row in dense_df.iterrows():
        # Vẽ Circle cho hành lang (giảm opacity để tạo hiệu ứng dải)
        if row['r6'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['r6']*1000, 
                          color=COL_R6, fill=True, weight=0, fill_opacity=0.1).add_to(m)
        if row['r10'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['r10']*1000, 
                          color=COL_R10, fill=True, weight=0, fill_opacity=0.15).add_to(m)
        if row['rc'] > 0:
            folium.Circle(location=[row['lat'], row['lon']], radius=row['rc']*1000, 
                          color=COL_RC, fill=True, weight=0, fill_opacity=0.2).add_to(m)

    # Vẽ các Marker điểm gốc (không nội suy) để người dùng click
    for _, row in raw_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']], radius=5, color="black", fill=True,
            popup=f"Cấp: {row.get('cường độ (cấp BF)', 'N/A')}"
        ).add_to(m)

    st_folium(m, width="100%", height=600)
    st.write("Dữ liệu sau khi nội suy dọc quỹ đạo (Densified Data):")
    st.dataframe(dense_df)
else:
    st.error("Vui lòng kiểm tra file besttrack.xlsx")
