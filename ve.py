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
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(page_title="Hệ thống Dự báo Bão - Phong Le", layout="wide")

# --- 1. HÀM TÍNH TOÁN & NỘI SUY (10KM) ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=1):
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

# --- 2. HÀM XUẤT ẢNH PNG (SỬA LỖI TẢI XUỐNG) ---
def get_png_map(df):
    plt.switch_backend('Agg') # Tránh lỗi giao diện trên server Linux
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.plot(df['lon'], df['lat'], color='black', marker='o', markersize=3, linewidth=1.5, label='Quỹ đạo bão')
    ax.set_title(f"Bản đồ Quỹ đạo Bão - {df.iloc[0].get('Ngày - giờ', '2026')}")
    ax.set_xlabel("Kinh độ (E)"); ax.set_ylabel("Vĩ độ (N)")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Thêm bảng nhỏ vào ảnh
    cell_text = df[['Ngày - giờ', 'lat', 'lon']].tail(5).values
    ax.table(cellText=cell_text, colLabels=['Thời gian', 'Vĩ độ', 'Kinh độ'], loc='bottom', bbox=[0, -0.3, 1, 0.2])
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# --- 3. HTML CHO BẢNG TIN & HỘP CÔNG CỤ (NẰM TRONG MAP) ---
def get_floating_ui_html(df):
    last_pts = df.tail(4)
    html = f"""
    <div style="position: fixed; top: 10px; right: 10px; width: 280px; z-index:9999; 
                background: white; padding: 10px; border: 2px solid #333; border-radius: 8px; 
                font-family: Arial; font-size: 11px; max-height: 400px; overflow-y: auto; opacity: 0.95;">
        <h4 style="margin: 0 0 8px 0; text-align: center; color: red;">HỘP CÔNG CỤ & TIN BÃO</h4>
        
        <p><b>Dữ liệu mới nhất:</b></p>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 10px;">
            <tr style="background: #eee;"><th>Giờ</th><th>Vị trí</th><th>Gió</th></tr>
    """
    for _, r in last_pts.iterrows():
        html += f"<tr><td style='border:1px solid #ccc;padding:3px;'>{r['Ngày - giờ']}</td>"
        html += f"<td style='border:1px solid #ccc;padding:3px;'>{r['lat']}N-{r['lon']}E</td>"
        html += f"<td style='border:1px solid #ccc;padding:3px;'>Cấp {int(r['cường độ (cấp BF)'])}</td></tr>"
    
    html += """
        </table>
        <hr>
        <p style="text-align:center; font-style: italic;">Sử dụng Sidebar bên trái để tải file PNG/Excel (Do bảo mật trình duyệt).</p>
    </div>
    """
    return html

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists(DATA_FILE):
    raw_df = pd.read_excel(DATA_FILE)
    raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['lat', 'lon'])
    dense_df = densify_track(raw_df, step_km=10)

    # SIDEBAR: Chỉ dùng để đặt nút Tải xuống (Folium không cho đặt nút tải trực tiếp trong map)
    with st.sidebar:
        st.header("💾 Tải Dữ Liệu")
        png_data = get_png_map(raw_df)
        st.download_button("🖼️ Tải ảnh bản đồ PNG", png_data, "storm_map.png", "image/png")
        st.download_button("📥 Tải Excel Dự báo", raw_df.to_csv(index=False).encode('utf-8'), "du_bao.csv")

    # KHỞI TẠO BẢN ĐỒ
    m = folium.Map(location=[17.0, 112.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Vẽ dải gió nội suy 10km
    for key, color, opacity in [('r6', COL_R6, 0.35), ('r10', COL_R10, 0.45), ('rc', COL_RC, 0.55)]:
        for _, row in dense_df.iterrows():
            if row[key] > 0:
                folium.Circle(location=[row['lat'], row['lon']], radius=row[key]*1000, 
                              color=color, fill=True, weight=0, fill_opacity=opacity).add_to(m)

    # 2. Vẽ đường quỹ đạo & Icon
    folium.PolyLine(raw_df[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
    for _, row in raw_df.iterrows():
        # (Sử dụng logic get_custom_icon đã viết ở bước trước của bạn)
        folium.CircleMarker([row['lat'], row['lon']], radius=3, color='black').add_to(m)

    # 3. Gắn UI nổi (Bảng tin & Hộp công cụ cuộn)
    m.get_root().html.add_child(folium.Element(get_floating_ui_html(raw_df)))
    
    # 4. Gắn Chú thích ảnh
    chuthich_img = os.path.join(ICON_DIR, "chuthich.PNG")
    if os.path.exists(chuthich_img):
        FloatImage(chuthich_img, bottom=5, left=2).add_to(m)

    st_folium(m, width="100%", height=700)
else:
    st.error("Thiếu file besttrack.xlsx")
