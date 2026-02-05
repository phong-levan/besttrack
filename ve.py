# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import io
import base64
from math import radians, sin, cos, asin, sqrt

# Thư viện hình học để xử lý "khoét lỗ" vùng gió
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

# --- CẤU HÌNH HỆ THỐNG ---
ICON_DIR = "icon"
DATA_FILE = "besttrack.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG") 
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90" 

st.set_page_config(page_title="Hệ thống Theo dõi Bão - Phong Le", layout="wide")

# --- CSS INJECTION: FIX CỨNG MÀN HÌNH, TRÀN VIỀN ---
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden;
        height: 100vh;
        width: 100vw;
    }
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        height: 100vh !important;
    }
    header, footer, #MainMenu {visibility: hidden;}
    iframe {
        height: 100vh !important;
        width: 100vw !important;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CÁC HÀM HỖ TRỢ ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=10):
    new_rows = []
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
    if os.path.exists(path):
        return folium.CustomIcon(path, icon_size=(35, 35) if bf >= 8 else (22, 22))
    return None

# --- 2. BẢNG THÔNG TIN DỰ BÁO (THEO MẪU ẢNH) ---
def get_info_table_html(df):
    # Lọc chỉ lấy dữ liệu dự báo
    f_df = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)]
    
    rows_html = ""
    for _, r in f_df.iterrows():
        rows_html += f"""
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="padding: 5px; border-right: 1px solid #ddd;">{r['Ngày - giờ']}</td>
            <td style="padding: 5px; border-right: 1px solid #ddd;">{float(r['lon']):.1f}E</td>
            <td style="padding: 5px; border-right: 1px solid #ddd;">{float(r['lat']):.1f}N</td>
            <td style="padding: 5px; border-right: 1px solid #ddd;">Cấp {int(r['cường độ (cấp BF)'])}</td>
            <td style="padding: 5px;">{int(r.get('Pmin (mb)', 0))}</td>
        </tr>
        """
    
    # Tiêu đề bảng lấy từ điểm dự báo cuối cùng hoặc mặc định
    table_html = f"""
    <div style="position: fixed; top: 15px; left: 60px; width: 420px; z-index: 9999; 
                background: rgba(255,255,255,0.9); padding: 10px; border: 2px solid #000; 
                border-radius: 5px; font-family: 'Arial Black', sans-serif; font-size: 11px;
                box-shadow: 3px 3px 10px rgba(0,0,0,0.3);">
        <div style="text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 2px;">TIN VỀ CƠN BÃO</div>
        <div style="text-align: center; font-size: 12px; margin-bottom: 8px;">Tin phát lúc: {f_df.iloc[0]['Ngày - giờ'] if not f_df.empty else ""}</div>
        <table style="width: 100%; border-collapse: collapse; text-align: center;">
            <tr style="background: #f8f9fa; font-weight: bold; border-bottom: 2px solid #333;">
                <th style="padding: 5px; border-right: 1px solid #ddd;">Ngày - giờ</th>
                <th style="padding: 5px; border-right: 1px solid #ddd;">Kinh độ</th>
                <th style="padding: 5px; border-right: 1px solid #ddd;">Vĩ độ</th>
                <th style="padding: 5px; border-right: 1px solid #ddd;">Cấp gió</th>
                <th style="padding: 5px;">Pmin(mb)</th>
            </tr>
            {rows_html}
        </table>
    </div>
    """
    return table_html

# --- 3. LOGIC HÌNH HỌC & HIỂN THỊ ---
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
    final_rc = uc
    final_r10 = u10.difference(uc) if u10 and uc else u10
    final_r6 = u6.difference(u10) if u6 and u10 else u6
    return final_r6, final_r10, final_rc

if os.path.exists(DATA_FILE):
    raw_df = pd.read_excel(DATA_FILE)
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])
    dense_df = densify_track(raw_df, step_km=10)

    m = folium.Map(location=[17.0, 115.0], zoom_start=5, tiles="OpenStreetMap", control_scale=True)

    f6, f10, fc = create_storm_swaths(dense_df)
    for geom, color, opacity in [(f6, COL_R6, 0.5), (f10, COL_R10, 0.6), (fc, COL_RC, 0.7)]:
        if geom and not geom.is_empty:
            folium.GeoJson(mapping(geom), style_function=lambda x, c=color, o=opacity: {'fillColor': c, 'color': c, 'weight': 1, 'fillOpacity': o}).add_to(m)

    folium.PolyLine(raw_df[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
    for _, row in raw_df.iterrows():
        icon = get_storm_icon(row)
        if icon: folium.Marker([row['lat'], row['lon']], icon=icon).add_to(m)

    # --- CHÈN BẢNG THÔNG TIN DỰ BÁO ---
    m.get_root().html.add_child(folium.Element(get_info_table_html(raw_df)))

    # --- CHÚ THÍCH (GÓC TRÊN PHẢI) ---
    if os.path.exists(CHUTHICH_IMG):
        with open(CHUTHICH_IMG, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        legend_html = f'''<div style="position: fixed; top: 15px; right: 15px; width: 35%; max-width: 380px; z-index: 9999; pointer-events: none;"><img src="data:image/png;base64,{encoded}" style="width: 100%;"></div>'''
        m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=2000, use_container_width=True)
else:
    st.error("Thiếu file besttrack.xlsx")
