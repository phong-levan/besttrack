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

# Mã màu chuẩn từ nghiên cứu của Phong
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90" 
COL_TRACK = "black"

st.set_page_config(page_title="Hệ thống Dự báo Bão - Phong Le", layout="wide")

# --- 1. TIỆN ÍCH TÍNH TOÁN & NỘI SUY ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(phi1)*cos(phi2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=10):
    """Nội suy dọc quỹ đạo với bước 10km để tạo dải gió mịn"""
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

# --- 2. QUẢN LÝ ICON & DỮ LIỆU LỊCH SỬ ---
def get_custom_icon(row):
    status = "daqua" if "quá khứ" in str(row.get('Thời điểm', '')).lower() else "dubao"
    bf = row.get('cường độ (cấp BF)', 0)
    if pd.isna(bf) or bf < 6: fname = f"vungthap{status}.png"
    elif bf < 8: fname = "atnddaqua.PNG" if status == "daqua" else "atnd.PNG"
    elif bf <= 11: fname = "bnddaqua.PNG" if status == "daqua" else "bnd.PNG"
    else: fname = "sieubaodaqua.PNG" if status == "daqua" else "sieubao.PNG"
    
    path = os.path.join(ICON_DIR, fname)
    if os.path.exists(path):
        return folium.CustomIcon(path, icon_size=(35, 35) if bf >= 8 else (20, 20))
    return None

def update_history_file(df):
    past_df = df[df['Thời điểm'].str.contains("quá khứ", case=False, na=False)].copy()
    if os.path.exists(HISTORY_FILE):
        old = pd.read_excel(HISTORY_FILE)
        df_hist = pd.concat([old, past_df]).drop_duplicates(subset=['Ngày - giờ'])
        df_hist.to_excel(HISTORY_FILE, index=False)
    else:
        past_df.to_excel(HISTORY_FILE, index=False)

# --- 3. GIAO DIỆN BẢNG THÔNG TIN (HTML) ---
def get_floating_table_html(df):
    last_pts = df.tail(5) # Hiển thị 5 điểm mới nhất
    html = f"""
    <div style="position: fixed; top: 15px; right: 15px; width: 300px; z-index:9999; 
                background: white; padding: 12px; border: 2px solid #333; border-radius: 8px; 
                font-family: 'Segoe UI', Arial; font-size: 11px; box-shadow: 3px 3px 10px rgba(0,0,0,0.2);">
        <h4 style="margin: 0 0 10px 0; text-align: center; color: #d32f2f;">BẢNG TIN BÃO KHẨN CẤP</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #eee;">
                <th style="border: 1px solid #ccc; padding: 4px;">Ngày-Giờ</th>
                <th style="border: 1px solid #ccc; padding: 4px;">Vị trí</th>
                <th style="border: 1px solid #ccc; padding: 4px;">Gió</th>
            </tr>
    """
    for _, r in last_pts.iterrows():
        html += f"<tr><td style='border: 1px solid #ccc; padding: 4px;'>{r['Ngày - giờ']}</td>"
        html += f"<td style='border: 1px solid #ccc; padding: 4px;'>{r['lat']}N-{r['lon']}E</td>"
        html += f"<td style='border: 1px solid #ccc; padding: 4px;'>Cấp {int(r['cường độ (cấp BF)'])}</td></tr>"
    html += "</table></div>"
    return html

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists(DATA_FILE):
    df = pd.read_excel(DATA_FILE)
    df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    
    update_history_file(df)
    dense_df = densify_track(df, step_km=10)

    # SIDEBAR CÔNG CỤ
    with st.sidebar:
        st.header("🛠️ Hộp Công Cụ")
        st.download_button("📥 Xuất Excel Dự báo", df.to_csv(index=False).encode('utf-8'), "du_bao.csv")
        if st.button("🖼️ Tạo ảnh bản đồ PNG"):
            st.info("Tính năng đang khởi tạo dựa trên Matplotlib...")

    # KHỞI TẠO BẢN ĐỒ
    st.subheader("🌀 Bản đồ Quỹ đạo Bão & Hành lang Gió Tương tác")
    m = folium.Map(location=[17.0, 112.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Vẽ hành lang gió xếp lớp (Hồng -> Đỏ -> Xanh)
    for key, color, opacity in [('r6', COL_R6, 0.35), ('r10', COL_R10, 0.45), ('rc', COL_RC, 0.55)]:
        for _, row in dense_df.iterrows():
            if row[key] > 0:
                folium.Circle(location=[row['lat'], row['lon']], radius=row[key]*1000, 
                              color=color, fill=True, weight=0, fill_opacity=opacity).add_to(m)

    # 2. Vẽ đường quỹ đạo chính
    folium.PolyLine(df[['lat', 'lon']].values.tolist(), color=COL_TRACK, weight=2).add_to(m)

    # 3. Thêm Icon tùy chỉnh tại các điểm tâm bão
    for _, row in df.iterrows():
        st_icon = get_custom_icon(row)
        if st_icon:
            folium.Marker(location=[row['lat'], row['lon']], icon=st_icon,
                          popup=f"{row['Ngày - giờ']}: Cấp {row['cường độ (cấp BF)']}").add_to(m)

    # 4. Gắn bảng thông tin và chú thích vào bên trong bản đồ
    m.get_root().html.add_child(folium.Element(get_floating_table_html(df)))
    
    chuthich_img = os.path.join(ICON_DIR, "chuthich.PNG")
    if os.path.exists(chuthich_img):
        FloatImage(chuthich_img, bottom=5, left=2).add_to(m)

    st_folium(m, width="100%", height=700)
else:
    st.error("Lỗi: Không tìm thấy file 'besttrack.xlsx'. Hãy kiểm tra thư mục GitHub của bạn.")
