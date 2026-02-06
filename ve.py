# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
from math import radians, sin, cos, asin, sqrt

# Thư viện hình học để xử lý vùng gió
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

# --- 1. CẤU HÌNH HỆ THỐNG ---
ICON_DIR = "icon"
DATA_FILE = "besttrack.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG") 
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90" 

st.set_page_config(
    page_title="Hệ thống Theo dõi Bão", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. CSS INJECTION: FIX LỖI TRẮNG MÀN HÌNH & TRÀN VIỀN ---
st.markdown("""
    <style>
    /* Xóa bỏ hoàn toàn thanh cuộn và lề trình duyệt */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        overflow: hidden !important;
        height: 100vh !important;
        width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Xóa padding của container chính */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        height: 100vh !important;
    }

    /* Ẩn Header và Footer */
    [data-testid="stHeader"], footer {
        display: none !important;
    }
    
    /* Cấu hình Iframe bản đồ lấp đầy màn hình */
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
    }
    
    /* Đảm bảo Sidebar nằm trên */
    [data-testid="stSidebar"] {
        z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CÁC HÀM HỖ TRỢ ---
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

def get_right_dashboard_html(df, img_base64):
    current_df = df[df['Thời điểm'].str.contains("hiện tại", case=False, na=False)]
    forecast_df = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)]
    display_df = pd.concat([current_df, forecast_df])
    
    rows_html = "".join([f"""
        <tr style="border: 1px solid black; background: #ffffff;">
            <td style="border: 1px solid black; padding: 4px;">{r['Ngày - giờ']}</td>
            <td style="border: 1px solid black; padding: 4px;">{float(r['lon']):.1f}E</td>
            <td style="border: 1px solid black; padding: 4px;">{float(r['lat']):.1f}N</td>
            <td style="border: 1px solid black; padding: 4px;">Cấp {int(r['cường độ (cấp BF)'])}</td>
            <td style="border: 1px solid black; padding: 4px;">{int(r.get('Pmin (mb)', 0))}</td>
        </tr>""" for _, r in display_df.iterrows()])
    
    return f"""
    <div style="position: fixed; top: 15px; right: 15px; width: 350px; z-index: 9999; pointer-events: auto;">
        <img src="data:image/png;base64,{img_base64}" style="width: 100%; border-radius: 5px; margin-bottom: 8px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);">
        <div style="background: rgba(255,255,255,0.9); border: 2px solid #333; border-radius: 5px; padding: 8px; font-family: Arial, sans-serif;">
            <div style="text-align: center; font-size: 13px; font-weight: bold; margin-bottom: 5px; color: black;">TIN BÃO TRÊN BIỂN ĐÔNG</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 11px; text-align: center; color: black;">
                <tr style="background: #e0e0e0; border: 1px solid black;">
                    <th style="border: 1px solid black;">Giờ</th><th style="border: 1px solid black;">Kinh</th>
                    <th style="border: 1px solid black;">Vĩ</th><th style="border: 1px solid black;">Cấp</th>
                    <th style="border: 1px solid black;">Pmin</th>
                </tr>
                {rows_html}
            </table>
        </div>
    </div>"""

# --- 4. LOGIC HIỂN THỊ ---
if os.path.exists(DATA_FILE):
    raw_df = pd.read_excel(DATA_FILE)
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])

    # Sidebar chọn bão
    storm_col = 'Số hiệu' if 'Số hiệu' in raw_df.columns else None
    selected_storms = []
    if storm_col:
        st.sidebar.markdown("### 🌪️ BẬT / TẮT BÃO")
        unique_storms = raw_df[storm_col].unique()
        for s in unique_storms:
            if st.sidebar.checkbox(f"Bão số {s}", value=True):
                selected_storms.append(s)
        final_df = raw_df[raw_df[storm_col].isin(selected_storms)]
    else:
        final_df = raw_df

    # Khởi tạo bản đồ với các đường lưới kinh vĩ độ
    m = folium.Map(location=[17.5, 114.0], zoom_start=6, tiles="OpenStreetMap")

    # Vẽ lưới kinh vĩ độ (Graticule)
    for lon in range(100, 141, 5):
        folium.PolyLine([[0, lon], [40, lon]], color='gray', weight=0.5, opacity=0.4).add_to(m)
    for lat in range(0, 41, 5):
        folium.PolyLine([[lat, 100], [lat, 140]], color='gray', weight=0.5, opacity=0.4).add_to(m)

    if not final_df.empty:
        groups = [None] if not storm_col else selected_storms
        for storm_id in groups:
            storm_data = final_df[final_df[storm_col] == storm_id] if storm_col else final_df
            if storm_data.empty: continue
            
            dense_df = densify_track(storm_data)
            f6, f10, fc = create_storm_swaths(dense_df)
            for geom, col, op in [(f6, COL_R6, 0.4), (f10, COL_R10, 0.5), (fc, COL_RC, 0.6)]:
                if geom and not geom.is_empty:
                    folium.GeoJson(mapping(geom), style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(m)
            
            folium.PolyLine(storm_data[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
            for _, row in storm_data.iterrows():
                icon = get_storm_icon(row)
                if icon: folium.Marker([row['lat'], row['lon']], icon=icon).add_to(m)

        if os.path.exists(CHUTHICH_IMG):
            with open(CHUTHICH_IMG, "rb") as f:
                encoded_img = base64.b64encode(f.read()).decode()
            m.get_root().html.add_child(folium.Element(get_right_dashboard_html(final_df, encoded_img)))

    # HIỂN THỊ: Đặt chiều cao cố định lớn để ép Iframe hiện ra, CSS sẽ lo phần tràn viền
    st_folium(m, width=2500, height=1200, use_container_width=True)

else:
    st.error("Không tìm thấy file besttrack.xlsx")
