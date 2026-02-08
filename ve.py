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
from math import radians, sin, cos, asin, sqrt, pi
import warnings
import textwrap

# Thư viện hình học
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.csv"
FILE_OPT2 = "besttrack_capgio.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# --- ĐỊNH NGHĨA ICON PATHS ---
ICON_PATHS = {
    "vungthap_daqua": os.path.join(ICON_DIR, 'vungthapdaqua.png'),
    "atnd_daqua": os.path.join(ICON_DIR, 'atnddaqua.PNG'),
    "bnd_daqua": os.path.join(ICON_DIR, 'bnddaqua.PNG'),
    "sieubao_daqua": os.path.join(ICON_DIR, 'sieubaodaqua.PNG'),
    "vungthap_dubao": os.path.join(ICON_DIR, 'vungthapdubao.png'),
    "atnd_dubao": os.path.join(ICON_DIR, 'atnd.PNG'),
    "bnd_dubao": os.path.join(ICON_DIR, 'bnd.PNG'),
    "sieubao_dubao": os.path.join(ICON_DIR, 'sieubao.PNG')
}

# --- DANH SÁCH LINK WEB ---
LINK_WEATHEROBS = "https://weatherobs.com/"
LINK_WIND_AUTO = "https://kttvtudong.net/kttv"
LINK_KMA_FORECAST = "https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm=2026.02.06.12&delta=000&ftm=2026.02.06.12"

# Màu sắc
COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
COLOR_TEXT = "#333333"
COLOR_ACCENT = "#007bff"
COLOR_BORDER = "#dee2e6"
SIDEBAR_WIDTH = "320px"

# Cấu hình trang
st.set_page_config(
    page_title="Storm Monitor",
    layout="wide",
    initial_sidebar_state="expanded" 
)
# ==============================================================================
# 2. CSS CHUNG (CHIA TÁCH ĐỘC LẬP SIDEBAR VÀ NỘI DUNG)
# ==============================================================================
st.markdown(f"""
    <style>
    /* 1. THIẾT LẬP FLEXBOX CHO TOÀN BỘ APP */
    [data-testid="stAppViewContainer"] {{
        display: flex !important;
        flex-direction: row !important;
        width: 100vw !important;
        height: 100vh !important;
        overflow: hidden !important;
    }}

    /* 2. SIDEBAR BÊN TRÁI: CHIẾM CỐ ĐỊNH 320PX */
    section[data-testid="stSidebar"] {{
        position: relative !important; /* Không dùng fixed để nó giữ chỗ */
        width: {SIDEBAR_WIDTH} !important;
        min-width: {SIDEBAR_WIDTH} !important;
        max-width: {SIDEBAR_WIDTH} !important;
        height: 100vh !important;
        background-color: {COLOR_SIDEBAR} !important;
        border-right: 1px solid #ddd;
        margin: 0 !important;
        z-index: 2 !important;
    }}

    /* 3. PHẦN NỘI DUNG CHÍNH: CHIẾM TOÀN BỘ PHẦN CÒN LẠI */
    [data-testid="stMainViewContainer"] {{
        flex: 1 !important; /* Tự động lấp đầy phần màn hình từ 320px trở đi */
        display: flex !important;
        flex-direction: column !important;
        min-width: 0 !important;
        height: 100vh !important;
        background-color: white !important;
        overflow: hidden !important;
    }}

    /* XÓA CÁC LỚP ĐỆM CỦA STREAMLIT ĐỂ IFRAME TRÀN VIỀN */
    .block-container {{
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        height: 100% !important;
        max-width: 100% !important;
    }}
    
    header, footer {{
        display: none !important;
    }}

    /* ẨN NÚT ĐÓNG/MỞ SIDEBAR */
    [data-testid="stSidebarCollapseBtn"],
    [data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}

    /* 4. ÉP IFRAME PHẢI VỪA KHÍT TRONG PHẦN CÒN LẠI */
    iframe {{
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
    }}

    /* 5. CÁC WIDGET NỔI TRÊN BẢN ĐỒ */
    .legend-box {{
        position: fixed; top: 10px; right: 10px; z-index: 1000;
        width: 280px; pointer-events: none;
    }}
    .info-box {{
        position: fixed; top: 220px; right: 10px; z-index: 1000;
        width: fit-content; background: rgba(255, 255, 255, 0.9);
        border: 1px solid #ccc; border-radius: 6px;
        padding: 8px !important; color: #000;
    }}
    </style>
""", unsafe_allow_html=True)
# ==============================================================================
# 3. HÀM XỬ LÝ LOGIC
# ==============================================================================

@st.cache_data(ttl=300) 
def get_rainviewer_ts():
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        r = requests.get(url, timeout=3, verify=False)
        return r.json()['satellite']['infrared'][-1]['time']
    except: return None

def image_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    ext = image_path.split('.')[-1].lower()
    mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {
        "tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no",
        "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "khí áp": "pressure", "khí áp (mb)": "pressure", "pmin": "pressure", "pmin (mb)": "pressure"
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

def generate_circle_polygon(lat, lon, radius_km, n_points=36):
    coords = []
    if radius_km <= 0: return None
    lat_rad = radians(lat)
    for i in range(n_points):
        theta = (i / n_points) * (2 * pi)
        dy = (radius_km * cos(theta)) / 111.32
        dx = (radius_km * sin(theta)) / (111.32 * cos(lat_rad))
        coords.append((lon + dx, lat + dy))
    return Polygon(coords)

def create_storm_swaths(dense_df):
    polys = {'r6': [], 'r10': [], 'rc': []}
    for _, row in dense_df.iterrows():
        for r, key in [(row.get('r6',0), 'r6'), (row.get('r10',0), 'r10'), (row.get('rc',0), 'rc')]:
            if r > 0:
                poly = generate_circle_polygon(row['lat'], row['lon'], r)
                if poly: polys[key].append(poly)
    u = {k: unary_union(v) if v else None for k, v in polys.items()}
    f_rc = u['rc']
    f_r10 = u['r10'].difference(u['rc']) if u['r10'] and u['rc'] else u['r10']
    f_r6 = u['r6'].difference(u['r10']) if u['r6'] and u['r10'] else u['r6']
    return f_r6, f_r10, f_rc

def get_icon_name(row):
    wind_speed = row.get('bf', 0) 
    w = row.get('wind_km/h', 0)
    
    if pd.isna(wind_speed) or wind_speed == 0:
        if w > 0:
            if w < 34: wind_speed = 5
            elif w < 64: wind_speed = 7
            elif w < 100: wind_speed = 10
            else: wind_speed = 12
    
    status_raw = str(row.get('status_raw','')).lower()
    
    status = 'dubao' 
    if 'quá khứ' in status_raw or 'past' in status_raw:
        status = 'daqua'
    
    if pd.isna(wind_speed): return f"vungthap_{status}"
    if wind_speed < 6:      return f"vungthap_{status}"
    if wind_speed < 8:      return f"atnd_{status}"
    if wind_speed <= 11:    return f"bnd_{status}"
    return f"sieubao_{status}"

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
        w = r.get('wind_km/h', 0)
        
        lon = f"{r.get('lon', 0):.1f}E"
        lat = f"{r.get('lat', 0):.1f}N"
        
        bf = r.get('bf', 0)
        if (pd.isna(bf) or bf == 0) and w > 0:
             if w < 34: bf = 6
             elif w < 64: bf = 8
             elif w < 100: bf = 10
             else: bf = 12
        cap_gio = f"Cấp {int(bf)}" if bf > 0 else "-"
        
        p = r.get('pressure', 0)
        pmin = f"{int(p)}" if (pd.notna(p) and p > 0) else "-"

        rows += f"<tr><td>{t}</td><td>{lon}</td><td>{lat}</td><td>{cap_gio}</td><td>{pmin}</td></tr>"
    
    return textwrap.dedent(f"""
    <div class="info-box">
        <div class="info-title">{title}</div>
        <div class="info-subtitle">(Dữ liệu cập nhật từ Besttrack)</div>
        <table>
            <thead>
                <tr>
                    <th>Ngày-Giờ</th>
                    <th>Kinh độ</th>
                    <th>Vĩ độ</th>
                    <th>Cấp gió</th>
                    <th>Pmin</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""")

def create_legend(img_b64):
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="legend-box">
        <img src="data:image/png;base64,{img_b64}">
    </div>""")

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    
    with st.sidebar:
        st.title("🌪️ TRUNG TÂM BÃO")
        
        topic = st.radio("CHỌN CHẾ ĐỘ:", 
                         ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""
        
        # Khởi tạo mặc định để tránh lỗi Syntax
        obs_mode = ""

        if topic == "Dữ liệu quan trắc":
            obs_mode = st.radio("Chọn nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)"])

        if topic == "Bản đồ Bão":
            storm_opt = st.selectbox("Dữ liệu bão:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])
            active_mode = storm_opt
            
            if "Hiện trạng" in storm_opt:
                dashboard_title = "TIN BÃO KHẨN CẤP"
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack.csv", type="csv", key="o1")
                    path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                    
                    def process_file(f_path):
                        if not f_path: return pd.DataFrame()
                        try:
                            if isinstance(f_path, str):
                                if f_path.endswith('.csv'): df = pd.read_csv(f_path)
                                else: df = pd.read_excel(f_path)
                            else: 
                                if f_path.name.endswith('.csv'): df = pd.read_csv(f_path)
                                else: df = pd.read_excel(f_path)
                                
                            df = normalize_columns(df)
                            if 'name' not in df.columns and 'storm_no' not in df.columns:
                                df['name'] = 'Cơn bão'
                                df['storm_no'] = 'Current Storm'

                            for c in ['wind_km/h', 'bf', 'r6', 'r10', 'rc', 'pressure']: 
                                if c not in df.columns: df[c] = 0
                            if 'datetime_str' in df.columns: df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
                            elif all(c in df.columns for c in ['year','mon','day','hour']): df['dt'] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day, hour=df.hour), errors='coerce')
                            for c in ['lat','lon','wind_km/h', 'pressure', 'bf']: df[c] = pd.to_numeric(df[c], errors='coerce')
                            return df.dropna(subset=['lat','lon'])
                        except: return pd.DataFrame()

                    df = process_file(path)
                    if not df.empty:
                        if 'storm_no' in df.columns:
                            all_s = df['storm_no'].unique()
                            sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s)
                            final_df = df[df['storm_no'].isin(sel)]
                        else:
                            final_df = df
                    else: st.warning("Vui lòng tải file.")
            else: 
                # (Phần lịch sử giữ nguyên)
                dashboard_title = "THỐNG KÊ LỊCH SỬ"
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack_capgio.xlsx", type="xlsx", key="o2")
                    path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
                    if path:
                        try:
                            df = pd.read_excel(path)
                            df = normalize_columns(df)
                            for c in ['wind_km/h', 'bf', 'r6', 'r10', 'rc', 'pressure']: 
                                if c not in df.columns: df[c] = 0
                            if 'datetime_str' in df.columns: df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
                            elif all(c in df.columns for c in ['year','mon','day','hour']): df['dt'] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day, hour=df.hour), errors='coerce')
                            for c in ['lat','lon','wind_km/h', 'pressure']: df[c] = pd.to_numeric(df[c], errors='coerce')
                            df = df.dropna(subset=['lat','lon'])

                            years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                            temp = df[df['year'].isin(years)]
                            names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                            final_df = temp[temp['name'].isin(names)]
                        except: pass
                    else: st.warning("Vui lòng tải file.")

    # --- MAIN CONTENT ---
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=1000&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1")
    elif topic == "Dữ liệu quan trắc":
        if "WeatherObs" in obs_mode:
            components.iframe(LINK_WEATHEROBS, scrolling=True)
        elif "Gió tự động" in obs_mode:
            components.iframe(LINK_WIND_AUTO, scrolling=True)
    elif topic == "Dự báo điểm (KMA)":
        components.iframe(LINK_KMA_FORECAST, scrolling=True)
    elif topic == "Bản đồ Bão":
        m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
        folium.TileLayer('CartoDB positron', name='Bản đồ Sáng (Mặc định)', overlay=False, control=True).add_to(m)
        folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết', overlay=False, control=True).add_to(m)
        folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Vệ tinh (Nền)', overlay=False, control=True).add_to(m)
        
        ts = get_rainviewer_ts()
        if ts: folium.TileLayer(tiles=f"https://tile.rainviewer.com/{ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png", attr="RainViewer", name="☁️ Mây Vệ tinh", overlay=True, show=True, opacity=0.5).add_to(m)

        fg_storm = folium.FeatureGroup(name="🌀 Đường đi Bão")
        if not final_df.empty and show_widgets:
            if "Hiện trạng" in str(active_mode):
                groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
                for g in groups:
                    sub = final_df[final_df['storm_no']==g] if g else final_df
                    dense = densify_track(sub)
                    f6, f10, fc = create_storm_swaths(dense)
                    for geom, c, o in [(f6,'#FFC0CB',0.4), (f10,'#FF6347',0.5), (fc,'#90EE90',0.6)]:
                        if geom and not geom.is_empty: folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg_storm)
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_storm)
                    
                    # --- VẼ ICON BÃO ---
                    for _, r in sub.iterrows():
                        icon_key = get_icon_name(r)
                        icon_path = ICON_PATHS.get(icon_key)
                        icon_base64 = None
                        if icon_path:
                            icon_base64 = image_to_base64(icon_path)
                        
                        if icon_base64:
                            icon = folium.CustomIcon(icon_image=icon_base64, icon_size=(40, 40), icon_anchor=(20, 20))
                            folium.Marker(location=[r['lat'], r['lon']], icon=icon, tooltip=f"Gió: {r.get('wind_km/h', 0)} km/h").add_to(fg_storm)
            else: 
                for n in final_df['name'].unique():
                    sub = final_df[final_df['name']==n].sort_values('dt')
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg_storm)
                    for _, r in sub.iterrows():
                        c = '#00f2ff' if r.get('wind_km/h',0)<64 else '#ff0055'
                        folium.CircleMarker([r['lat'],r['lon']], radius=3, color=c, fill=True, popup=f"{n}").add_to(fg_storm)
        
        fg_storm.add_to(m)
        folium.LayerControl(position='topleft', collapsed=False).add_to(m)
        
        if show_widgets:
            if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG):
                with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                st.markdown(create_legend(b64), unsafe_allow_html=True)
            
            if not final_df.empty: 
                st.markdown(create_info_table(final_df, dashboard_title), unsafe_allow_html=True)
            else: 
                st.markdown(create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU..."), unsafe_allow_html=True)
        
        st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()


