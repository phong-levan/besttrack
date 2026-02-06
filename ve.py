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
from math import radians, sin, cos, asin, sqrt
import warnings
import textwrap

# Thư viện xử lý hình học
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & GIAO DIỆN "CYBERPUNK"
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"
FILE_OPT2 = "besttrack_capgio.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# Bảng màu Neon cho giao diện tối
COLOR_ACCENT = "#00f2ff" # Cyan Neon
COLOR_WARNING = "#ff0055" # Neon Red
COLOR_BG_DARK = "rgba(16, 22, 35, 0.95)"

st.set_page_config(
    page_title="Storm & Satellite Monitor", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🌪️"
)

# --- CSS CAO CẤP: GIAO DIỆN TỐI & GLASSMORPHISM ---
st.markdown(f"""
    <style>
    /* Xóa nền mặc định, ép nền tối */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: #0e1117 !important;
    }}
    
    /* Sidebar đẹp hơn */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_BG_DARK} !important;
        border-right: 1px solid #333;
    }}
    [data-testid="stSidebar"] h1 {{
        color: {COLOR_ACCENT} !important;
        font-family: 'Helvetica Neue', sans-serif;
        text-transform: uppercase;
        font-size: 24px;
    }}
    
    /* Bản đồ full màn hình */
    iframe {{ position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; z-index: 0; }}
    
    /* INFO BOX: Hiệu ứng kính mờ (Glassmorphism) */
    .info-box {{
        z-index: 9999 !important;
        font-family: 'Segoe UI', sans-serif;
        background: rgba(20, 20, 30, 0.85); /* Nền đen mờ */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        color: white;
        overflow: hidden;
    }}
    
    /* Bảng dữ liệu trong Dashboard */
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{ background: linear-gradient(90deg, #004e92, #000428); color: {COLOR_ACCENT}; padding: 8px; text-transform: uppercase; letter-spacing: 1px; }}
    td {{ padding: 6px; border-bottom: 1px solid #333; text-align: center; color: #ddd; }}
    tr:hover {{ background-color: rgba(0, 242, 255, 0.1); }}
    
    /* Layer Control Custom */
    .leaflet-control-layers {{
        background: {COLOR_BG_DARK} !important;
        color: white !important;
        border: 1px solid {COLOR_ACCENT} !important;
        border-radius: 8px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HÀM XỬ LÝ DỮ LIỆU (ROBUST)
# ==============================================================================

@st.cache_data(ttl=300) 
def get_rainviewer_ts():
    """Lấy TS RainViewer (Update 10p/lần)"""
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        r = requests.get(url, timeout=3, verify=False)
        data = r.json()
        if 'satellite' in data and 'infrared' in data['satellite']:
            return data['satellite']['infrared'][-1]['time']
    except: return None
    return None

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        "tên bão": "name", "name": "name", "biển đông": "storm_no", "số hiệu": "storm_no",
        "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "vĩ độ": "lat", "kinh độ": "lon", "gió (kt)": "wind_kt", "khí áp (mb)": "pressure",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
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
# 3. UI COMPONENTS (DASHBOARD ĐẸP)
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
    <div class="info-box" style="position: fixed; top: 20px; right: 20px; width: 320px;">
        <div style="background: linear-gradient(90deg, #ff0055, #ff00cc); padding: 10px; font-weight: bold; text-align: center; letter-spacing: 1px;">{title}</div>
        {content}
    </div>""")

def create_legend(img_b64):
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; bottom: 30px; right: 20px; width: 280px; padding: 10px;">
        <div style="text-align:center; font-weight:bold; font-size:12px; margin-bottom:8px; color: #00f2ff;">CHÚ GIẢI KÝ HIỆU</div>
        <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:4px; border: 1px solid #444;">
    </div>""")

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    with st.sidebar:
        st.title("🌪️ STORM MONITOR")
        st.caption("Real-time Satellite & Tracking System")
        
        # --- CHỌN CHẾ ĐỘ ---
        topic = st.radio("CHẾ ĐỘ HIỂN THỊ:", ["Bản đồ Bão (Storm Map)", "Vệ tinh Windy (Satellite)"], index=0)
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""

        # --- HÀM ĐỌC FILE ---
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

        # === LOGIC KHI Ở CHẾ ĐỘ BẢN ĐỒ BÃO ===
        if topic == "Bản đồ Bão (Storm Map)":
            storm_opt = st.selectbox("Dữ liệu bão:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])
            active_mode = storm_opt
            
            if "Hiện trạng" in storm_opt:
                dashboard_title = "TIN BÃO KHẨN CẤP"
                show_layer = st.checkbox("Hiển thị lớp Dữ liệu", value=True)
                if show_layer:
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack.xlsx", type="xlsx", key="o1")
                    path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                    df = process_excel(path)
                    if not df.empty:
                        if 'storm_no' in df.columns:
                            all_s = df['storm_no'].unique()
                            sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s)
                            final_df = df[df['storm_no'].isin(sel)]
                        else: final_df = df
                    else: st.warning("Vui lòng tải file dữ liệu.")

            else: # Lịch sử
                dashboard_title = "THỐNG KÊ LỊCH SỬ"
                show_layer = st.checkbox("Hiển thị lớp Dữ liệu", value=True)
                if show_layer:
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack_capgio.xlsx", type="xlsx", key="o2")
                    path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
                    df = process_excel(path)
                    if not df.empty:
                        years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                        temp = df[df['year'].isin(years)]
                        names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                        final_df = temp[temp['name'].isin(names)]
                    else: st.warning("Vui lòng tải file dữ liệu.")

        # === LOGIC KHI Ở CHẾ ĐỘ WINDY ===
        elif topic == "Vệ tinh Windy (Satellite)":
            st.success("✅ Đang kết nối máy chủ Windy...")
            st.info("Dữ liệu: Vệ tinh Hồng ngoại (Satellite INFRA+)")
            # URL Embed Windy chuẩn, tắt menu thừa, bật lớp Satellite
            windy_url = "https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=800&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
            components.iframe(windy_url, height=1000, scrolling=False)
            return # Dừng render phần map bên dưới

    # --- KHỞI TẠO BẢN ĐỒ FOLIUM (GIAO DIỆN TỐI) ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
    
    # 1. Lớp Nền "Dark Matter" (Đẹp và chuyên nghiệp cho bão)
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB', name='Bản đồ Tối (Dark)', overlay=False, control=True
    ).add_to(m)
    
    # 2. Lớp Vệ tinh Google (Cho ai thích nhìn thực địa)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite', name='Vệ tinh (Google)', overlay=False, control=True
    ).add_to(m)

    # 3. LỚP MÂY VỆ TINH REAL-TIME (RAINVIEWER) - OVERLAY
    latest_ts = get_rainviewer_ts()
    if latest_ts:
        folium.TileLayer(
            tiles=f"https://tile.rainviewer.com/{latest_ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
            attr="RainViewer", name="☁️ Mây Vệ tinh (Real-time)", overlay=True, show=True, opacity=0.5
        ).add_to(m)

    fg_storm = folium.FeatureGroup(name="🌀 Đường đi Bão")
    
    # 4. VẼ BÃO (MÀU SẮC NEON)
    if not final_df.empty and topic == "Bản đồ Bão (Storm Map)" and show_widgets:
        if "Hiện trạng" in str(active_mode):
            groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
            for g in groups:
                sub = final_df[final_df['storm_no']==g] if g else final_df
                dense = densify_track(sub)
                f6, f10, fc = create_storm_swaths(dense)
                
                # Vẽ vùng gió (trong suốt nhẹ)
                for geom, c, o in [(f6,'#ff00ff',0.3), (f10,'#ff0055',0.4), (fc,'#00f2ff',0.5)]:
                    if geom and not geom.is_empty: folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg_storm)
                
                # Đường đi màu trắng sáng
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='white', weight=2, dash_array='5').add_to(fg_storm)
                
                # Icon
                for _, r in sub.iterrows():
                    icon_path = os.path.join(ICON_DIR, f"{get_icon_name(r)}.png")
                    if os.path.exists(icon_path): folium.Marker([r['lat'],r['lon']], icon=folium.CustomIcon(icon_path, icon_size=(35,35))).add_to(fg_storm)
                    else: folium.CircleMarker([r['lat'],r['lon']], radius=4, color=COLOR_ACCENT, fill=True).add_to(fg_storm)
        else: # Lịch sử
            for n in final_df['name'].unique():
                sub = final_df[final_df['name']==n].sort_values('dt')
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color=COLOR_ACCENT, weight=1.5, opacity=0.8).add_to(fg_storm)
                for _, r in sub.iterrows():
                    c = '#00f2ff' if r.get('wind_kt',0)<64 else '#ff0055'
                    folium.CircleMarker([r['lat'],r['lon']], radius=3, color=c, fill=True, popup=f"{n}").add_to(fg_storm)

    fg_storm.add_to(m)
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)

    # 5. RENDER WIDGETS
    if show_widgets:
        if not final_df.empty: st.markdown(create_info_table(final_df, dashboard_title), unsafe_allow_html=True)
        else: st.markdown(create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU..."), unsafe_allow_html=True)
        
        if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG):
            with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()
            st.markdown(create_legend(b64), unsafe_allow_html=True)

    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
