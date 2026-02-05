# -*- coding: utf-8 -*-
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

# Import thư viện bản đồ chuyên nghiệp để xuất ảnh có nền
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# --- CẤU HÌNH HỆ THỐNG ---
ICON_DIR = "icon"
DATA_FILE = "besttrack.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# Mã màu chuyên dụng cho R6 (hồng), R10 (đỏ), RC (xanh)
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90" 

st.set_page_config(page_title="Hệ thống Dự báo Bão - Phong Le", layout="wide")

# --- 1. TIỆN ÍCH NỘI SUY (BƯỚC 10KM) ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=10):
    """Nội suy dọc quỹ đạo để tạo dải gió mịn màng cho nghiên cứu"""
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

# --- 2. QUẢN LÝ ICON BÃO ---
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

# --- 3. BẢNG TIN DỰ BÁO LƠ LỬNG (HIỂN THỊ HẾT DỰ BÁO) ---
def get_forecast_dashboard_html(df):
    # LỌC: Lấy TOÀN BỘ dữ liệu dự báo
    f_df = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)]
    
    rows_html = ""
    for _, r in f_df.iterrows():
        rows_html += f"""
        <tr>
            <td style="border:1px solid #ccc; padding:4px;">{r['Ngày - giờ']}</td>
            <td style="border:1px solid #ccc; padding:4px;">{r['lat']}N-{r['lon']}E</td>
            <td style="border:1px solid #ccc; padding:4px;">Cấp {int(r['cường độ (cấp BF)'])}</td>
            <td style="border:1px solid #ccc; padding:4px;">{int(r.get('Vmax (km/h)', 0))}</td>
            <td style="border:1px solid #ccc; padding:4px;">{int(r.get('Pmin (mb)', 0))}</td>
        </tr>"""
    
    return f"""
    <div style="position: fixed; top: 20px; right: 20px; width: 380px; z-index:9999; 
                background: rgba(255,255,255,0.95); padding: 15px; border: 2px solid #d32f2f; 
                border-radius: 10px; font-family: Arial; font-size: 11px; max-height: 450px; overflow-y: auto;
                box-shadow: 4px 4px 15px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 10px 0; text-align: center; color: #d32f2f; font-weight: bold;">TIN DỰ BÁO BÃO CHI TIẾT</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #d32f2f; color: white;">
                <th>Giờ</th><th>Tọa độ</th><th>Cấp</th><th>Vmax(km)</th><th>Pmin</th>
            </tr>
            {rows_html}
        </table>
    </div>"""

# --- 4. XUẤT ẢNH PNG CÓ NỀN BẢN ĐỒ ĐỊA LÝ (Dùng Cartopy) ---
def export_storm_png(df):
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(12, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Thiết lập phạm vi bản đồ dựa trên dữ liệu
    ax.set_extent([df['lon'].min()-5, df['lon'].max()+5, df['lat'].min()-5, df['lat'].max()+5])
    
    # Thêm nền bản đồ địa lý thực tế
    ax.stock_img() 
    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.7, zorder=3)
    
    # Vẽ quỹ đạo bão
    ax.plot(df['lon'], df['lat'], 'k-o', markersize=4, linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
    
    # Thêm bảng dữ liệu tọa độ vào dưới cùng của ảnh
    data = df[['Ngày - giờ', 'lat', 'lon', 'cường độ (cấp BF)', 'Pmin (mb)']].tail(8).values
    table = ax.table(cellText=data, colLabels=['Giờ', 'Vĩ độ', 'Kinh độ', 'Cấp', 'Pmin'], 
                     loc='bottom', bbox=[0, -0.3, 1, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
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

    with st.sidebar:
        st.header("💾 Hộp Công Cụ Xuất Bản")
        if HAS_CARTOPY:
            st.download_button("🖼️ Tải bản đồ PNG (Có nền địa lý)", export_storm_png(df), "bao_report_pro.png", "image/png")
        else:
            st.warning("⚠️ Đang cấu hình hệ thống Cartopy trên máy chủ...")
        st.download_button("📥 Tải dữ liệu Excel", df.to_csv(index=False).encode('utf-8'), "du_bao_bao.csv")

    # KHỞI TẠO BẢN ĐỒ TƯƠNG TÁC
    st.subheader(f"🌀 Hệ thống Theo dõi Bão Tương tác - Cập nhật: {df.iloc[-1].get('Ngày - giờ', '')}")
    m = folium.Map(location=[17.5, 115.0], zoom_start=5, tiles="OpenStreetMap")

    # 1. Vẽ hành lang gió nội suy (Trong suốt xếp lớp)
    for k, c, o in [('r6', COL_R6, 0.4), ('r10', COL_R10, 0.5), ('rc', COL_RC, 0.6)]:
        for _, row in dense_df.iterrows():
            if row[k] > 0:
                folium.Circle([row['lat'], row['lon']], radius=row[k]*1000, color=c, fill=True, weight=0, fill_opacity=o).add_to(m)

    # 2. Vẽ đường đi và Icon bão
    folium.PolyLine(df[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
    for _, row in df.iterrows():
        icon = get_storm_icon(row)
        if icon:
            folium.Marker([row['lat'], row['lon']], icon=icon, popup=f"{row['Ngày - giờ']}").add_to(m)

    # 3. Gắn bảng tin dự báo và Chú thích vào Map
    m.get_root().html.add_child(folium.Element(get_forecast_dashboard_html(df)))
    if os.path.exists(CHUTHICH_IMG):
        FloatImage(CHUTHICH_IMG, bottom=5, left=2).add_to(m)

    st_folium(m, width="100%", height=750)
else:
    st.error("Lỗi: Không tìm thấy file besttrack.xlsx")
