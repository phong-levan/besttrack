# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
import requests
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from math import radians, sin, cos, asin, sqrt
import warnings
import textwrap

# Thư viện hình học
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & GIAO DIỆN LIGHT MODE (TRẮNG)
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"
FILE_OPT2 = "besttrack_capgio.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# Thông tin đăng nhập Web vệ tinh riêng
TARGET_URL = "http://222.255.11.82/Default.aspx"
TARGET_USER = "admin"
TARGET_PASS = "ttdl@2021"

# --- BẢNG MÀU SÁNG (LIGHT THEME) ---
COLOR_BG = "#ffffff"          # Nền chính: Trắng
COLOR_SIDEBAR = "#f8f9fa"     # Sidebar: Xám rất nhạt (chuẩn Bootstrap)
COLOR_TEXT = "#333333"        # Chữ: Đen xám
COLOR_ACCENT = "#007bff"      # Điểm nhấn: Xanh dương (Blue)
COLOR_BORDER = "#dee2e6"      # Viền: Xám nhạt

st.set_page_config(
    page_title="Storm Monitor Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH CHO GIAO DIỆN TRẮNG ---
st.markdown(f"""
    <style>
    /* 1. Nền ứng dụng chính */
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
    }}
    
    /* 2. Sidebar (Thanh bên trái) */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_SIDEBAR} !important;
        border-right: 1px solid {COLOR_BORDER};
    }}
    /* Chữ trong Sidebar */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {COLOR_TEXT} !important;
    }}
    
    /* 3. Info Box (Bảng tin nổi) - Nền trắng bóng đổ nhẹ */
    .info-box {{
        z-index: 9999;
        font-family: 'Segoe UI', sans-serif;
        background: rgba(255, 255, 255, 0.95); /* Trắng đục */
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        color: {COLOR_TEXT};
        box-shadow: 0 4px 12px rgba(0,0,0,0.15); /* Bóng đổ mềm */
    }}
    
    /* 4. Table trong Dashboard */
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ 
        background-color: {COLOR_ACCENT}; 
        color: white; 
        padding: 8px; 
        text-transform: uppercase; 
        font-weight: 600;
    }}
    td {{ 
        padding: 6px; 
        border-bottom: 1px solid {COLOR_BORDER}; 
        text-align: center; 
        color: {COLOR_TEXT}; 
    }}
    tr:nth-child(even) {{ background-color: #f2f2f2; }} /* Dòng chẵn màu xám nhẹ */
    
    /* 5. Layer Control (Hộp công cụ bản đồ) */
    .leaflet-control-layers {{
        background: white !important;
        color: {COLOR_TEXT} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 8px !important;
        padding: 10px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    }}
    .leaflet-control-layers-expanded::before {{
        content: "🛠️ HỘP CÔNG CỤ";
        display: block; font-weight: bold; text-align: center; color: {COLOR_ACCENT}; 
        margin-bottom: 5px; border-bottom: 1px solid #eee;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ (GIỮ NGUYÊN LOGIC, CHỈ SỬA GIAO DIỆN)
# ==============================================================================

@st.cache_data(ttl=600)
def login_and_fetch_web(url, username, password):
    session = requests.Session()
    try:
        r1 = session.get(url, timeout=10)
        soup = BeautifulSoup(r1.text, 'html.parser')
        payload = {tag['name']: tag.get('value', '') for tag in soup.find_all('input') if tag.get('name')}
        
        user_in = soup.find('input', {'type': 'text'})
        pass_in = soup.find('input', {'type': 'password'})
        if user_in and pass_in:
            payload[user_in['name']] = username
            payload[pass_in['name']] = password
            r2 = session.post(url, data=payload, timeout=15)
            content = r2.text.replace('<head>', f'<head><base href="{url}">')
            return content
        return "<h3>Không tìm thấy khung đăng nhập.</h3>"
    except Exception as e:
        return f"<h3>Lỗi kết nối: {e}</h3>"

@st.cache_data(ttl=300) 
def get_rainviewer_ts():
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        r = requests.get(url, timeout=3, verify=False)
        return r.json()['satellite']['infrared'][-1]['time']
    except: return None

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {
        "tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no",
        "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "vĩ độ": "lat", "kinh độ": "lon", "gió (kt)": "wind_kt",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc"
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    return df

def densify_track(df, step_km=10):
    new_rows = []
    if len(df) < 2: return df
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = 6371 * 2 * asin(sqrt(sin(radians(p2['lat']-p1['lat'])/2)**2 + cos(radians(p1['lat']))*cos(radians(p2['lat']))*sin(radians(p2['lon']-p1['lon'])/2)**2))
        steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(steps):
            f = j / steps
            row = p1.copy()
            row['lat'] = p1['lat'] + (p2['lat'] - p1['lat']) * f
            row['lon'] = p1['lon'] + (p2['lon'] - p1['lon']) * f
            for col in ['r6', 'r10', 'rc']:
                if col in p1 and col in p2: row[col] = p1.get(col, 0)*(1-f) + p2.get(col, 0)*f
            new_rows.append(row)
    new_rows.append(df.iloc[-1])
    return pd.DataFrame(new_rows)

def create_storm_swaths(dense_df):
    polys = {'r6': [], 'r10': [], 'rc': []}
    geo = geodesic.Geodesic()
    for _, row in dense_df.iterrows():
        for r, key in [(row.get('r6',0), 'r6'), (row.get('r10',0), 'r10'), (row.get('rc',0), 'rc')]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                polys[key].append(Polygon(circle))
    u = {k: unary_union(v) if v else None for k, v in polys.items()}
    f_rc = u['rc']
    f_r10 = u['r10'].difference(u['rc']) if u['r10'] and u['rc'] else u['r10']
    f_r6 = u['r6'].difference(u['r10']) if u['r6'] and u['r10'] else u['r6']
    return f_r6, f_r10, f_rc

def get_icon_name(row):
    w = row.get('wind_kt', 0)
    bf = row.get('bf', 0)
    if pd.isna(bf) or bf == 0:
        if w < 34: bf = 6
        elif w < 64: bf = 8
        elif w < 100: bf = 10
        else: bf = 12
    status = 'dubao' if 'forecast' in str(row.get('status_raw','')).lower() else 'daqua'
    if bf < 6: return f"vungthap_{status}"
    if bf < 8: return f"atnd_{status}"
    if bf <= 11: return f"bnd_{status}"
    return f"sieubao_{status}"

# ==============================================================================
# 3. UI COMPONENTS (STYLE TRẮNG)
# ==============================================================================

def create_info_table(df, title):
    if df.empty: return ""
    if 'status_raw' in df.columns:
         cur = df[df['status_raw'].astype(str).str.contains("hiện tại|current", case=False, na=False)]
         fut = df[df['status_raw'].astype(str).str.contains("dự báo|forecast", case=False, na=False)]
         display_df = pd.concat([cur, fut]).head(8)
    else:
         display_df = df.sort_values('dt', ascending=False).groupby('name').head(1)

    rows = ""
    for _, r in display_df.iterrows():
        t = r.get('datetime_str', r.get('dt'))
        if not isinstance(t, str): t = t.strftime('%d/%m %Hh')
        w = r.get('wind_kt', 0)
        rows += f"<tr><td>{t}</td><td>{r.get('lat',0):.1f}/{r.get('lon',0):.1f}</td><td>{int(w) if pd.notna(w) else 0}</td></tr>"
    
    content = f"<table><thead><tr><th>Thời gian</th><th>Vị trí</th><th>Gió (kt)</th></tr></thead><tbody>{rows}</tbody></table>"
    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; top: 10px; right: 10px; width: 320px;">
        <div style="background-color: {COLOR_ACCENT}; color: white; padding: 10px; font-weight: bold; text-align: center; border-radius: 8px 8px 0 0;">{title}</div>
        <div style="padding: 0;">{content}</div>
    </div>""")

def create_legend(img_b64):
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; bottom: 20px; right: 10px; width: 280px; padding: 10px;">
        <div style="text-align:center; font-weight:bold; font-size:12px; margin-bottom:8px; color: {COLOR_ACCENT};">CHÚ GIẢI KÝ HIỆU</div>
        <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:4px; border: 1px solid #ddd;">
    </div>""")

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    with st.sidebar:
        st.title("🌪️ TRUNG TÂM BÃO")
        st.caption("Phiên bản giao diện sáng")
        
        topic = st.radio("CHỌN CHẾ ĐỘ:", 
                         ["Bản đồ Bão (Storm Map)", "Vệ tinh (Windy)", "Vệ tinh (Private)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""

        def process_excel(f_path):
            if not f_path or not os.path.exists(f_path): return pd.DataFrame()
            try:
                df = pd.read_excel(f_path)
                df = normalize_columns(df)
                for c in ['wind_kt', 'bf', 'r6', 'r10', 'rc']: 
                    if c not in df.columns: df[c] = 0
                if 'datetime_str' in df.columns: df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
                elif all(c in df.columns for c in ['year','mon','day','hour']): df['dt'] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day, hour=df.hour), errors='coerce')
                for c in ['lat','lon','wind_kt']: df[c] = pd.to_numeric(df[c], errors='coerce')
                return df.dropna(subset=['lat','lon'])
            except: return pd.DataFrame()

        # === 1. BẢN ĐỒ BÃO ===
        if topic == "Bản đồ Bão (Storm Map)":
            storm_opt = st.selectbox("Dữ liệu:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])
            active_mode = storm_opt
            
            if "Hiện trạng" in storm_opt:
                dashboard_title = "TIN BÃO KHẨN CẤP"
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack.xlsx", type="xlsx", key="o1")
                    path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                    df = process_excel(path)
                    if not df.empty:
                        all_s = df['storm_no'].unique() if 'storm_no' in df.columns else []
                        sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s)
                        final_df = df[df['storm_no'].isin(sel)] if 'storm_no' in df.columns else df
                    else: st.warning("Vui lòng tải file.")

            else: # Lịch sử
                dashboard_title = "THỐNG KÊ LỊCH SỬ"
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack_capgio.xlsx", type="xlsx", key="o2")
                    path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
                    df = process_excel(path)
                    if not df.empty:
                        years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                        temp = df[df['year'].isin(years)]
                        names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                        final_df = temp[temp['name'].isin(names)]
                    else: st.warning("Vui lòng tải file.")

        # === 2. VỆ TINH WINDY ===
        elif topic == "Vệ tinh (Windy)":
            st.success("✅ Kết nối Windy (Real-time)...")
            components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=1000&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1", height=1000)
            return 

        # === 3. VỆ TINH NỘI BỘ (AUTO-LOGIN) ===
        elif topic == "Vệ tinh (Private)":
            st.warning(f"🔐 Auto-login: {TARGET_URL}...")
            with st.spinner("Đang xác thực..."):
                html = login_and_fetch_web(TARGET_URL, TARGET_USER, TARGET_PASS)
            components.html(html, height=1000, scrolling=True)
            return

    # --- RENDER BẢN ĐỒ FOLIUM (GIAO DIỆN SÁNG) ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
    
    # Lớp Nền Sáng (Positron) làm mặc định cho giao diện trắng
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng (Mặc định)', overlay=False, control=True).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết', overlay=False, control=True).add_to(m)
    # Vẫn giữ lớp vệ tinh để người dùng chuyển đổi nếu thích
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Vệ tinh (Nền)', overlay=False, control=True).add_to(m)

    # Lớp Mây RainViewer (Overlay)
    ts = get_rainviewer_ts()
    if ts: folium.TileLayer(tiles=f"https://tile.rainviewer.com/{ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png", attr="RainViewer", name="☁️ Mây Vệ tinh", overlay=True, show=True, opacity=0.5).add_to(m)

    # Vẽ Bão
    fg_storm = folium.FeatureGroup(name="🌀 Đường đi Bão")
    if not final_df.empty and topic == "Bản đồ Bão (Storm Map)" and show_widgets:
        if "Hiện trạng" in str(active_mode):
            groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
            for g in groups:
                sub = final_df[final_df['storm_no']==g] if g else final_df
                dense = densify_track(sub)
                f6, f10, fc = create_storm_swaths(dense)
                # Màu sắc vùng gió cho nền sáng: Rõ ràng hơn (Đỏ, Cam, Vàng)
                for geom, c, o in [(f6,'#FFC0CB',0.4), (f10,'#FF6347',0.5), (fc,'#DC143C',0.6)]:
                    if geom and not geom.is_empty: folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg_storm)
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_storm)
                for _, r in sub.iterrows():
                    icon_path = os.path.join(ICON_DIR, f"{get_icon_name(r)}.png")
                    if os.path.exists(icon_path): folium.Marker([r['lat'],r['lon']], icon=folium.CustomIcon(icon_path, icon_size=(35,35))).add_to(fg_storm)
                    else: folium.CircleMarker([r['lat'],r['lon']], radius=4, color='red', fill=True).add_to(fg_storm)
        else: 
            for n in final_df['name'].unique():
                sub = final_df[final_df['name']==n].sort_values('dt')
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg_storm)
                for _, r in sub.iterrows():
                    c = '#00f2ff' if r.get('wind_kt',0)<64 else '#ff0055'
                    folium.CircleMarker([r['lat'],r['lon']], radius=3, color=c, fill=True, popup=f"{n}").add_to(fg_storm)

    fg_storm.add_to(m)
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)

    # Widgets
    if show_widgets:
        if not final_df.empty: st.markdown(create_info_table(final_df, dashboard_title), unsafe_allow_html=True)
        else: st.markdown(create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU..."), unsafe_allow_html=True)
        if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG):
            with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()
            st.markdown(create_legend(b64), unsafe_allow_html=True)

    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
