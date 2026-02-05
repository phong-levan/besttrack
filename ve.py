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
CHUTHICH_FILE = os.path.join(ICON_DIR, "chuthich.PNG")

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

# --- 2. GIAO DIỆN BẢNG TIN ĐỘNG NỔI TRÊN MAP ---
def get_dynamic_table_html(df):
    # Lấy 5 thời điểm dự báo mới nhất
    last_pts = df.tail(5)
    
    rows_html = ""
    for _, r in last_pts.iterrows():
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
                background: white; padding: 12px; border: 2px solid #d32f2f; border-radius: 8px; 
                font-family: Arial; font-size: 11px; box-shadow: 4px 4px 12px rgba(0,0,0,0.3); opacity: 0.95;">
        <h4 style="margin: 0 0 10px 0; text-align: center; color: #d32f2f; font-weight: bold;">TIN BÃO TRÊN BIỂN ĐÔNG</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #d32f2f; color: white; text-align: center;">
                <th style="padding: 5px;">Giờ</th><th style="padding: 5px;">Vị trí</th>
                <th style="padding: 5px;">Cấp</th><th style="padding: 5px;">Gió(km)</th><th style="padding: 5px;">Áp suất</th>
            </tr>
            {rows_html}
        </table>
    </div>
    """
    return html

# --- 3. XUẤT ẢNH PNG CÓ CHÚ THÍCH & BẢNG (Dùng Matplotlib) ---
def get_static_png(df):
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    ax.plot(df['lon'], df['lat'], 'k-o', markersize=3, linewidth=1.5)
    ax.set_title(f"Bản đồ Quỹ đạo Bão - Trích xuất từ Hệ thống Phong Le")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Vẽ bảng thông tin ở chân ảnh giống yêu cầu của Phong
    data = df[['Ngày - giờ', 'lat', 'lon', 'cường độ (cấp BF)', 'Pmin (mb)']].tail(5).values
    ax.table(cellText=data, colLabels=['Giờ', 'Vĩ độ', 'Kinh độ', 'Cấp', 'Pmin'], 
             loc='bottom', bbox=[0, -0.3, 1, 0.2])
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists(DATA_FILE):
    df = pd.read_excel(DATA_FILE)
    df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    dense_df = densify_track(df, step_km=10)

    # SIDEBAR CÔNG CỤ TẢI XUỐNG
    with st.sidebar:
        st.header("💾 Tải Xuất Dữ Liệu")
        st.download_button("🖼️ Tải ảnh bản đồ PNG", get_static_png(df), "storm_report.png", "image/png")
        st.download_button("📥 Tải Excel Dự báo", df.to_csv(index=False).encode('utf-8'), "du_bao_bao.csv")

    # KHỞI TẠO MAP
    st.subheader(f"🌀 Theo dõi xoáy thuận nhiệt đới - {df.iloc[-1].get('Ngày - giờ', '2026')}")
    m = folium.Map(location=[16.5, 114.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Vẽ dải gió nội suy trong suốt (Hồng -> Đỏ -> Xanh)
    for key, color, op in [('r6', COL_R6, 0.4), ('r10', COL_R10, 0.5), ('rc', COL_RC, 0.6)]:
        for _, row in dense_df.iterrows():
            if row[key] > 0:
                folium.Circle(location=[row['lat'], row['lon']], radius=row[key]*1000, 
                              color=color, fill=True, weight=0, fill_opacity=op).add_to(m)

    # 2. Vẽ quỹ đạo và Icon
    folium.PolyLine(df[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
    # (Tự động chèn Icon từ logic get_custom_icon của bạn tại đây)

    # 3. GẮN CÁC THÀNH PHẦN NỔI CỐ ĐỊNH
    # Bảng thông tin động
    m.get_root().html.add_child(folium.Element(get_dynamic_table_html(df)))
    
    # Chú thích cố định từ thư mục icon
    if os.path.exists(CHUTHICH_FILE):
        FloatImage(CHUTHICH_FILE, bottom=5, left=2).add_to(m)

    st_folium(m, width="100%", height=750)
else:
    st.error("Không tìm thấy file besttrack.xlsx")
