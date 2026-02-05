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
    /* Hiển thị thanh sidebar chuyên nghiệp */
    [data-testid="stSidebar"] {
        background-color: #f1f3f4;
        border-right: 1px solid #d1d1d1;
    }
    header, footer, #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. CÁC HÀM HỖ TRỢ (GIỮ NGUYÊN) ---
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

# --- 2. THANH TÙY CHỌN (SIDEBAR) ---
with st.sidebar:
    st.header("🛠️ Công cụ dữ liệu")
    st.write("Tải dữ liệu bão hiện tại:")
    
    if os.path.exists(DATA_FILE):
        # Xuất file Excel
        df_to_export = pd.read_excel(DATA_FILE)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_to_export.to_excel(writer, index=False, sheet_name='BestTrack')
        
        st.download_button(
            label="📊 Xuất file Excel bão",
            data=output.getvalue(),
            file_name=f"BestTrack_{pd.Timestamp.now().strftime('%d%m%Y')}.xlsx",
            mime="application/vnd.ms-excel"
        )
    
    # Xuất ảnh PNG (Ảnh chú thích)
    if os.path.exists(CHUTHICH_IMG):
        with open(CHUTHICH_IMG, "rb") as f:
            st.download_button(
                label="🖼️ Tải ảnh chú thích (.png)",
                data=f,
                file_name="Chu_Thich_Bao.png",
                mime="image/png"
            )
    st.divider()
    st.info("Nhấn biểu tượng '>' ở góc trái để đóng/mở thanh công cụ.")

# --- 3. HÀM TẠO GIAO DIỆN KHỐI PHẢI (GIỮ NGUYÊN) ---
def get_right_dashboard_html(df, img_base64):
    current_df = df[df['Thời điểm'].str.contains("hiện tại", case=False, na=False)]
    forecast_df = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)]
    display_df = pd.concat([current_df, forecast_df])
    rows_html = ""
    for _, r in display_df.iterrows():
        rows_html += f"""
        <tr style="border: 1px solid black; background: #ffffff;">
            <td style="border: 1px solid black; padding: 4px;">{r['Ngày - giờ']}</td>
            <td style="border: 1px solid black; padding: 4px;">{float(r['lon']):.1f}E</td>
            <td style="border: 1px solid black; padding: 4px;">{float(r['lat']):.1f}N</td>
            <td style="border: 1px solid black; padding: 4px;">Cấp {int(r['cường độ (cấp BF)'])}</td>
            <td style="border: 1px solid black; padding: 4px;">{int(r.get('Pmin (mb)', 0))}</td>
        </tr>
        """
    dashboard_html = f"""
    <div style="position: fixed; top: 20px; right: 20px; width: 32%; max-width: 400px; z-index: 9999; pointer-events: auto;">
        <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto; margin-bottom: 10px;">
        <div style="background: rgba(255,255,255,0.95); border: 2px solid black; border-radius: 5px; padding: 8px; font-family: Arial, sans-serif; font-size: 11px;">
            <div style="text-align: center; font-size: 14px; font-weight: bold; color: black; margin-bottom: 5px; text-transform: uppercase;">
                Tin bão trên biển Đông
            </div>
            <table style="width: 100%; border-collapse: collapse; text-align: center; color: black; border: 1px solid black;">
                <thead>
                    <tr style="background: #e0e0e0; border: 1px solid black;">
                        <th style="border: 1px solid black; padding: 4px;">Giờ</th>
                        <th style="border: 1px solid black; padding: 4px;">Kinh độ</th>
                        <th style="border: 1px solid black; padding: 4px;">Vĩ độ</th>
                        <th style="border: 1px solid black; padding: 4px;">Cấp</th>
                        <th style="border: 1px solid black; padding: 4px;">Pmin</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """
    return dashboard_html

# --- 4. HIỂN THỊ BẢN ĐỒ ---
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

    if os.path.exists(CHUTHICH_IMG):
        with open(CHUTHICH_IMG, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode()
        m.get_root().html.add_child(folium.Element(get_right_dashboard_html(raw_df, encoded_img)))

    st_folium(m, width=None, height=2000, use_container_width=True)
else:
    st.error("Thiếu file besttrack.xlsx")
