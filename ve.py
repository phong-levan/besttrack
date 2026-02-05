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
# Đảm bảo tên file khớp chính xác với GitHub (phân biệt hoa thường)
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

# --- 2. GIAO DIỆN BẢNG TIN DỰ BÁO NỔI TRÊN MAP ---
def get_forecast_table_html(df):
    # LỌC: Chỉ lấy các điểm "dự báo"
    forecast_df = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)].copy()
    
    rows_html = ""
    for _, r in forecast_df.iterrows():
        rows_html += f"""
        <tr>
            <td style="border:1px solid #ccc; padding:4px;">{r['Ngày - giờ']}</td>
            <td style="border:1px solid #ccc; padding:4px;">{r['lat']}N-{r['lon']}E</td>
            <td style="border:1px solid #ccc; padding:4px;">Cấp {int(r['cường độ (cấp BF)'])}</td>
            <td style="border:1px solid #ccc; padding:4px;">{int(r.get('Vmax (km/h)', 0))}</td>
            <td style="border:1px solid #ccc; padding:4px;">{int(r.get('Pmin (mb)', 0))}</td>
        </tr>
        """
    
    # Nếu không có điểm dự báo nào
    if not rows_html:
        rows_html = "<tr><td colspan='5' style='text-align:center;'>Không có dữ liệu dự báo</td></tr>"

    html = f"""
    <div style="position: fixed; top: 15px; right: 15px; width: 380px; z-index:9999; 
                background: rgba(255,255,255,0.95); padding: 12px; border: 2px solid #d32f2f; 
                border-radius: 8px; font-family: Arial; font-size: 11px; box-shadow: 4px 4px 12px rgba(0,0,0,0.3);
                max-height: 400px; overflow-y: auto;">
        <h4 style="margin: 0 0 10px 0; text-align: center; color: #d32f2f; font-weight: bold;">TIN DỰ BÁO BÃO</h4>
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

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists(DATA_FILE):
    df = pd.read_excel(DATA_FILE)
    df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    dense_df = densify_track(df, step_km=10)

    # SIDEBAR CÔNG CỤ
    with st.sidebar:
        st.header("💾 Tải Xuất Dữ Liệu")
        st.download_button("📥 Tải Excel Dự báo", df.to_csv(index=False).encode('utf-8'), "du_bao_bao.csv")

    # KHỞI TẠO MAP
    st.subheader(f"🌀 Theo dõi xoáy thuận nhiệt đới - Cập nhật: {df.iloc[-1].get('Ngày - giờ', '')}")
    m = folium.Map(location=[17.5, 115.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Vẽ dải gió nội suy (Hồng -> Đỏ -> Xanh)
    for key, color, op in [('r6', COL_R6, 0.4), ('r10', COL_R10, 0.5), ('rc', COL_RC, 0.6)]:
        for _, row in dense_df.iterrows():
            if row[key] > 0:
                folium.Circle(location=[row['lat'], row['lon']], radius=row[key]*1000, 
                              color=color, fill=True, weight=0, fill_opacity=op).add_to(m)

    # 2. Vẽ quỹ đạo và Icon bão (Lấy từ thư mục icon/)
    folium.PolyLine(df[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
    # (Tại đây bạn thêm logic vòng lặp Marker với get_storm_icon của mình đã hướng dẫn trước đó)

    # 3. GẮN BẢNG TIN DỰ BÁO (Lọc bỏ quá khứ)
    m.get_root().html.add_child(folium.Element(get_forecast_table_html(df)))
    
    # 4. GẮN CHÚ THÍCH CỐ ĐỊNH
    # Kiểm tra kỹ file chuthich.PNG có tồn tại không để tránh lỗi
    if os.path.exists(CHUTHICH_FILE):
        # bottom=5, left=2 ghim ảnh ở góc dưới bên trái bản đồ
        FloatImage(CHUTHICH_FILE, bottom=5, left=2).add_to(m)
    else:
        st.sidebar.error(f"Không tìm thấy file: {CHUTHICH_FILE}")

    st_folium(m, width="100%", height=750)
else:
    st.error("Không tìm thấy file besttrack.xlsx")
