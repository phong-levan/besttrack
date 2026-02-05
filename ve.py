import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import io
import matplotlib.pyplot as plt
from math import radians, sin, cos, asin, sqrt
from folium.plugins import FloatImage

# --- CẤU HÌNH HỆ THỐNG ---
ICON_DIR = "icon"
DATA_FILE = "besttrack.xlsx"
HISTORY_FILE = "history_tracking.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# Mã màu chuyên dụng của Phong
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90" 

st.set_page_config(page_title="Hệ thống Theo dõi Bão - Phong Le", layout="wide")

# --- 1. TIỆN ÍCH TÍNH TOÁN & NỘI SUY (10KM) ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=10):
    new_rows = []
    for i in range(len(df)-1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine_km(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        n = max(1, int(np.ceil(dist/step_km)))
        for j in range(n):
            f = j/n
            new_rows.append({
                'lat': p1['lat'] + (p2['lat']-p1['lat'])*f,
                'lon': p1['lon'] + (p2['lon']-p1['lon'])*f,
                'r6': p1.get('bán kính gió mạnh cấp 6 (km)',0)*(1-f) + p2.get('bán kính gió mạnh cấp 6 (km)',0)*f,
                'r10': p1.get('bán kính gió mạnh cấp 10 (km)',0)*(1-f) + p2.get('bán kính gió mạnh cấp 10 (km)',0)*f,
                'rc': p1.get('bán kính tâm (km)',0)*(1-f) + p2.get('bán kính tâm (km)',0)*f
            })
    new_rows.append(df.iloc[-1].to_dict())
    return pd.DataFrame(new_rows)

# --- 2. XỬ LÝ BIỂU TƯỢNG (ICON) BÃO ---
def get_storm_icon(row):
    status = "daqua" if "quá khứ" in str(row.get('Thời điểm', '')).lower() else "dubao"
    bf = row.get('cường độ (cấp BF)', 0)
    
    # Khớp tên file với thư mục icon của Phong
    if pd.isna(bf) or bf < 6: fname = f"vungthap{status}.png"
    elif bf < 8: fname = "atnddaqua.PNG" if status == "daqua" else "atnd.PNG"
    elif bf <= 11: fname = "bnddaqua.PNG" if status == "daqua" else "bnd.PNG"
    else: fname = "sieubaodaqua.PNG" if status == "daqua" else "sieubao.PNG"
    
    path = os.path.join(ICON_DIR, fname)
    if os.path.exists(path):
        size = (35, 35) if bf >= 8 else (22, 22)
        return folium.CustomIcon(path, icon_size=size)
    return None

# --- 3. HTML CHO BẢNG TIN NỔI TRONG MAP (5 CỘT) ---
def get_floating_info_html(df):
    rows_html = ""
    # Lấy dữ liệu dự báo/hiện tại
    for _, r in df.iterrows():
        rows_html += f"""
        <tr>
            <td style="border:1px solid #ccc; padding:4px;">{r['Ngày - giờ']}</td>
            <td style="border:1px solid #ccc; padding:4px;">{r['lat']}N-{r['lon']}E</td>
            <td style="border:1px solid #ccc; padding:4px;">Cấp {int(r['cường độ (cấp BF)'])}</td>
            <td style="border:1px solid #ccc; padding:4px;">{int(r.get('Vmax (km/h)', 0))}</td>
            <td style="border:1px solid #ccc; padding:4px;">{int(r.get('Pmin (mb)', 0))}</td>
        </tr>
        """
    
    html = f"""
    <div style="position: fixed; top: 15px; right: 15px; width: 380px; z-index:9999; 
                background: rgba(255,255,255,0.9); padding: 12px; border: 2px solid #d32f2f; 
                border-radius: 8px; font-family: Arial; font-size: 11px; max-height: 450px; overflow-y: auto;">
        <h4 style="margin: 0 0 10px 0; text-align: center; color: #d32f2f;">BẢNG TIN & CÔNG CỤ</h4>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 10px;">
            <tr style="background: #d32f2f; color: white; text-align: center;">
                <th>Giờ</th><th>Vị trí</th><th>Cấp</th><th>Gió(km)</th><th>Áp suất</th>
            </tr>
            {rows_html}
        </table>
        <p style="text-align: center; font-style: italic; color: #555;">Sử dụng Sidebar bên trái để tải dữ liệu (PNG/Excel).</p>
    </div>
    """
    return html

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists(DATA_FILE):
    df = pd.read_excel(DATA_FILE)
    df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    dense_df = densify_track(df, step_km=10)

    # Sidebar cho các nút tải xuống (Do giới hạn bảo mật trình duyệt)
    with st.sidebar:
        st.header("💾 Tải Xuất Dữ Liệu")
        # Nút Excel
        st.download_button("📥 Tải Excel Dự báo", df.to_csv(index=False).encode('utf-8'), "du_bao_bao.csv")
        st.info("Để tải ảnh PNG có đầy đủ chú thích, hãy sử dụng tính năng 'Print' của trình duyệt hoặc nút chụp màn hình chuyên dụng.")

    # Khởi tạo bản đồ
    st.subheader(f"🌀 Hệ thống Theo dõi xoáy thuận nhiệt đới - {df.iloc[-1].get('Ngày - giờ', '')}")
    m = folium.Map(location=[16.5, 115.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Vẽ dải gió nội suy 10km xếp lớp trong suốt
    for key, color, op in [('r6', COL_R6, 0.4), ('r10', COL_R10, 0.5), ('rc', COL_RC, 0.6)]:
        for _, row in dense_df.iterrows():
            if row[key] > 0:
                folium.Circle(location=[row['lat'], row['lon']], radius=row[key]*1000, 
                              color=color, fill=True, weight=0, fill_opacity=op).add_to(m)

    # 2. Vẽ đường quỹ đạo & Icon
    folium.PolyLine(df[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
    for _, row in df.iterrows():
        icon = get_storm_icon(row)
        if icon:
            folium.Marker([row['lat'], row['lon']], icon=icon, 
                          popup=f"{row['Ngày - giờ']}: Cấp {row['cường độ (cấp BF)']}").add_to(m)

    # 3. Ghi bảng thông tin vào bản đồ
    m.get_root().html.add_child(folium.Element(get_floating_info_html(df)))
    
    # 4. Gắn chú thích cố định
    if os.path.exists(CHUTHICH_IMG):
        FloatImage(CHUTHICH_IMG, bottom=5, left=2).add_to(m)

    st_folium(m, width="100%", height=750)
else:
    st.error("Lỗi: Không tìm thấy file 'besttrack.xlsx'.")
