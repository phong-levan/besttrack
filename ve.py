import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
from math import radians, sin, cos, asin, sqrt

# --- CẤU HÌNH MÀU SẮC & ĐƯỜNG DẪN ---
ICON_DIR = "icon"
COL_R6   = "#FFC0CB"  # Hồng
COL_R10  = "#FF6347"  # Đỏ cam
COL_RC   = "#90EE90"  # Xanh lá
COL_TRACK = "black"

st.set_page_config(page_title="Hệ thống Theo dõi Bão - Phong Le", layout="wide")
st.title("🌀 Bản đồ Nội suy Quỹ đạo & Biểu tượng Bão")

# --- HÀM TÍNH KHOẢNG CÁCH ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# --- HÀM NỘI SUY (Dày đặc 10km để tạo dải mịn) ---
def densify_storm_data(df, step_km=0):
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

# --- HÀM LẤY ICON CHUẨN (Phân biệt hoa/thường) ---
def get_storm_icon(row):
    status = "daqua" if "quá khứ" in str(row.get('Thời điểm', '')).lower() else "dubao"
    bf = row.get('cường độ (cấp BF)', 0)
    
    if pd.isna(bf) or bf < 6:
        fname = f"vungthap{status}.png"
    elif bf < 8:
        fname = "atnddaqua.PNG" if status == "daqua" else "atnd.PNG"
    elif bf <= 11:
        fname = "bnddaqua.PNG" if status == "daqua" else "bnd.PNG"
    else:
        fname = "sieubaodaqua.PNG" if status == "daqua" else "sieubao.PNG"
    
    path = os.path.join(ICON_DIR, fname)
    if os.path.exists(path):
        size = (35, 35) if bf >= 8 else (20, 20)
        return folium.CustomIcon(path, icon_size=size)
    return None

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists("besttrack.xlsx"):
    raw_df = pd.read_excel("besttrack.xlsx")
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])
    dense_df = densify_storm_data(raw_df)

    m = folium.Map(location=[15.8, 112.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Vẽ hành lang gió trong suốt (3 lớp ngoài vào trong)
    for r_key, col, op in [('r6', COL_R6, 0.3), ('r10', COL_R10, 0.4), ('rc', COL_RC, 0.5)]:
        for _, row in dense_df.iterrows():
            if row[r_key] > 0:
                folium.Circle(location=[row['lat'], row['lon']], radius=row[r_key]*1000, 
                              color=col, fill=True, weight=0, fill_opacity=op).add_to(m)

    # 2. Vẽ đường quỹ đạo
    points = raw_df[['lat', 'lon']].values.tolist()
    if len(points) > 1:
        folium.PolyLine(points, color=COL_TRACK, weight=2, opacity=1).add_to(m)

    # 3. Vẽ Icon bão tại các điểm gốc
    for _, row in raw_df.iterrows():
        st_icon = get_storm_icon(row)
        if st_icon:
            folium.Marker(
                location=[row['lat'], row['lon']], icon=st_icon,
                popup=folium.Popup(f"Thời gian: {row.get('Ngày - giờ', 'N/A')}<br>Cấp: {row.get('cường độ (cấp BF)', 'N/A')}", max_width=200)
            ).add_to(m)
        else:
            folium.CircleMarker(location=[row['lat'], row['lon']], radius=4, color="red", fill=True).add_to(m)

    st_folium(m, width="100%", height=600)
else:
    st.error("Thiếu file besttrack.xlsx")

