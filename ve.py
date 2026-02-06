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
import time

# Thư viện xử lý hình học
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & HÀM HỖ TRỢ REAL-TIME
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"
FILE_OPT2 = "besttrack_capgio.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(page_title="Hệ thống Giám sát Bão Real-time", layout="wide", initial_sidebar_state="expanded")

# CSS GIAO DIỆN
st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: transparent !important; }
    header, footer { display: none !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; z-index: 0; }
    [data-testid="stSidebar"] { z-index: 10000 !important; background-color: rgba(28, 35, 49, 0.95) !important; }
    
    .leaflet-top.leaflet-left .leaflet-control-layers {
        background: rgba(255,255,255,0.95) !important;
        border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); padding: 10px; min-width: 200px;
    }
    .leaflet-control-layers-expanded::before {
        content: "🛠️ HỘP CÔNG CỤ"; display: block; font-weight: bold; text-align: center; color: #d63384; margin-bottom: 5px; border-bottom: 1px solid #eee;
    }
    .info-box { z-index: 9999 !important; font-family: Arial, sans-serif; }
    table { width: 100%; border-collapse: collapse; background: white; font-size: 11px; }
    td, th { padding: 4px; border: 1px solid #ddd; text-align: center; color: black; }
    th { background: #007bff; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- HÀM LẤY TIMESTAMP VỆ TINH (RAINVIEWER) ---
@st.cache_data(ttl=300) 
def get_rainviewer_ts():
    """Lấy TS RainViewer (Update mỗi 5-10 phút)"""
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        # Thêm verify=False để tránh lỗi SSL trong một số môi trường mạng
        response = requests.get(url, timeout=3, verify=False)
        data = response.json()
        if 'satellite' in data and 'infrared' in data['satellite']:
            return data['satellite']['infrared'][-1]['time']
    except Exception as e:
        return None
    return None

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU BÃO
# ==============================================================================
def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        "tên bão": "name", "name": "name", "biển đông": "storm_no", "storm_no": "storm_no", "số hiệu": "storm_no",
        "năm": "year", "tháng": "mon", "ngày": "day", "giờ": "hour",
        "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "vĩ độ": "lat", "kinh độ": "lon",
        "gió (kt)": "wind_kt", "khí áp (mb)": "pressure", "cường độ (cấp bf)": "bf",
        "bán kính gió mạnh cấp 6 (km)": "r6", "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=10):
    new_rows = []
    if len(df) < 2: return df
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine_km(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
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
# 3. HTML DASHBOARD
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
        if pd.isna(w): w = 0
        rows += f"<tr><td>{t}</td><td>{r.get('lat',0):.1f}/{r.get('lon',0):.1f}</td><td>{int(w)}</td></tr>"
    content = f"<table><thead><tr><th>Thời gian</th><th>Vị trí</th><th>Gió (kt)</th></tr></thead><tbody>{rows}</tbody></table>"
    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; top: 20px; right: 20px; width: 300px; background: white; border-radius: 8px; border: 1px solid #999; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="background:#007bff; color:white; padding:8px; text-align:center; font-weight:bold;">{title}</div>
        {content}
    </div>""")

def create_legend(img_b64):
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; bottom: 30px; right: 20px; width: 260px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; border: 1px solid #999; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="text-align:center; font-weight:bold; font-size:12px; margin-bottom:5px;">CHÚ GIẢI</div>
        <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:4px;">
    </div>""")

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    with st.sidebar:
        st.title("🎛️ ĐIỀU KHIỂN")
        
        # --- CẤU HÌNH REAL-TIME ---
        st.sidebar.markdown("### ⏱️ Cấu hình")
        auto_refresh = st.sidebar.checkbox("🔄 Tự động cập nhật (10p)", value=False)
        if auto_refresh:
            components.html("""<script>setTimeout(function(){window.location.reload();}, 600000);</script>""", height=0, width=0)

        st.markdown("---")
        topic = st.selectbox("1. CHỦ ĐỀ CHÍNH:", ["Bão (Typhoon)", "Thời tiết (Weather)"])
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

        # === NHÁNH 1: BÃO ===
        if topic == "Bão (Typhoon)":
            storm_opt = st.radio("2. CHỨC NĂNG:", ["Option 1: Hiện trạng", "Option 2: Lịch sử"])
            active_mode = storm_opt
            st.markdown("---")
            
            if "Option 1" in storm_opt:
                dashboard_title = "TIN BÃO HIỆN TẠI"
                if st.checkbox("Hiển thị lớp Hiện trạng", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack.xlsx", type="xlsx", key="o1")
                    path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                    df = process_excel(path)
                    if not df.empty:
                        if 'storm_no' in df.columns:
                            all_s = df['storm_no'].unique()
                            sel = st.multiselect("Chọn bão:", all_s, default=all_s)
                            final_df = df[df['storm_no'].isin(sel)]
                        else: final_df = df
                    else: st.warning("Vui lòng tải file.")

            else: # Option 2
                dashboard_title = "LỊCH SỬ BÃO"
                if st.checkbox("Hiển thị lớp Lịch sử", value=True):
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

        # === NHÁNH 2: THỜI TIẾT ===
        elif topic == "Thời tiết (Weather)":
            weather_source = st.radio("2. NGUỒN DỮ LIỆU:", ["Option 3: Quan trắc", "Option 4: Mô hình"])
            st.markdown("---")
            w_param = st.radio("3. THÔNG SỐ:", ["Nhiệt độ", "Lượng mưa", "Gió"])
            if st.checkbox("Hiển thị lớp dữ liệu", value=True):
                show_widgets = True
                dashboard_title = f"BẢN ĐỒ {str(w_param).upper()}"

    # --- KHỞI TẠO BẢN ĐỒ ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
    
    # 1. LỚP NỀN CƠ BẢN
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    # 2. LỚP VỆ TINH NỀN (ESRI) - Ổn định nhất
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='🛰️ Vệ tinh (Nền)', overlay=False
    ).add_to(m)

    # 3. LỚP MÂY VỆ TINH REAL-TIME (OVERLAY)
    # Lấy timestamp RainViewer
    latest_ts = get_rainviewer_ts()
    
    # HIỂN THỊ TRẠNG THÁI KẾT NỐI VỆ TINH TRONG SIDEBAR
    if latest_ts:
        st.sidebar.success(f"✅ Vệ tinh RainViewer: Online ({latest_ts})")
        # RainViewer Infrared
        folium.TileLayer(
            tiles=f"https://tile.rainviewer.com/{latest_ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
            attr="RainViewer",
            name="☁️ Mây Vệ tinh (RainViewer)",
            overlay=True, show=True, opacity=0.6 # Show mặc định
        ).add_to(m)
    else:
        st.sidebar.error("⚠️ Vệ tinh RainViewer: Offline (Dùng nguồn dự phòng)")
        # FALLBACK: RealEarth Global IR (Nguồn dự phòng rất mạnh)
        folium.TileLayer(
            tiles="https://realearth.ssec.wisc.edu/tiles/globalir/{z}/{x}/{y}.png",
            attr="RealEarth",
            name="☁️ Mây Vệ tinh (Global IR)",
            overlay=True, show=True, opacity=0.6
        ).add_to(m)

    fg_storm = folium.FeatureGroup(name="🌀 Lớp Bão")
    fg_weather = folium.FeatureGroup(name="🌦️ Lớp Thời Tiết")

    # 4. VẼ DỮ LIỆU BÃO
    if not final_df.empty and topic == "Bão (Typhoon)" and show_widgets:
        if "Option 1" in str(active_mode):
            groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
            for g in groups:
                sub = final_df[final_df['storm_no']==g] if g else final_df
                dense = densify_track(sub)
                f6, f10, fc = create_storm_swaths(dense)
                for geom, c, o in [(f6,COL_R6,0.4), (f10,COL_R10,0.5), (fc,COL_RC,0.6)]:
                    if geom and not geom.is_empty: folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':0,'fillOpacity':o}).add_to(fg_storm)
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_storm)
                for _, r in sub.iterrows():
                    icon_path = os.path.join(ICON_DIR, f"{get_icon_name(r)}.png")
                    if os.path.exists(icon_path): folium.Marker([r['lat'],r['lon']], icon=folium.CustomIcon(icon_path, icon_size=(30,30))).add_to(fg_storm)
                    else: folium.CircleMarker([r['lat'],r['lon']], radius=3, color='black').add_to(fg_storm)
        else: 
            for n in final_df['name'].unique():
                sub = final_df[final_df['name']==n].sort_values('dt')
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg_storm)
                for _, r in sub.iterrows():
                    c = '#00CCFF' if r.get('wind_kt',0)<34 else ('#FFFF00' if r.get('wind_kt',0)<64 else '#FF0000')
                    folium.CircleMarker([r['lat'],r['lon']], radius=4, color=c, fill=True, fill_opacity=1, popup=f"{n}").add_to(fg_storm)

    if topic == "Thời tiết (Weather)" and show_widgets:
        folium.Circle([16, 112], radius=100000, color='orange', fill=True, fill_opacity=0.3, popup="Vùng giả lập").add_to(fg_weather)

    fg_storm.add_to(m)
    fg_weather.add_to(m)
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)

    if show_widgets:
        if not final_df.empty: st.markdown(create_info_table(final_df, dashboard_title), unsafe_allow_html=True)
        elif topic == "Thời tiết (Weather)": st.markdown(create_info_table(pd.DataFrame(), dashboard_title), unsafe_allow_html=True)
        
        if "Option 1" in str(active_mode) and os.path.exists(CHUTHICH_IMG):
            with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()
            st.markdown(create_legend(b64), unsafe_allow_html=True)

    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
