# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
from math import radians, sin, cos, asin, sqrt
import warnings
import textwrap

# Thư viện xử lý hình học
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"        # Bão Hiện trạng
FILE_OPT2 = "besttrack_capgio.xlsx" # Bão Lịch sử
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(
    page_title="Hệ thống Tích hợp Bão & Thời tiết",
    layout="wide",
    initial_sidebar_state="expanded" # Luôn mở Sidebar để thấy các Ops
)

# CSS QUY HOẠCH GIAO DIỆN (FULL SCREEN & LAYER CONTROL TRÁI)
st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: transparent !important; }
    header, footer { display: none !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    
    /* Bản đồ full màn hình lớp dưới */
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; z-index: 0; }
    
    /* Sidebar lớp trên cùng */
    [data-testid="stSidebar"] { z-index: 10000 !important; background-color: rgba(28, 35, 49, 0.95) !important; }
    
    /* Layer Control góc TRÁI TRÊN giống Dashboard */
    .leaflet-top.leaflet-left .leaflet-control-layers {
        background: rgba(255,255,255,0.95) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
        padding: 10px !important;
        border: 1px solid #999 !important;
        min-width: 180px;
    }
    .leaflet-control-layers-expanded::before {
        content: "🛠️ CÔNG CỤ LỚP";
        display: block; font-weight: bold; text-align: center; color: #d63384; 
        margin-bottom: 5px; font-family: Arial; font-size: 12px; border-bottom: 1px solid #eee;
    }
    
    /* Info Box Style */
    .info-box { z-index: 9999 !important; font-family: Arial, sans-serif; }
    table { width: 100%; border-collapse: collapse; background: white; font-size: 11px; }
    td, th { padding: 4px; border: 1px solid #ddd; text-align: center; color: black; }
    th { background: #007bff; color: white; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ SỐ LIỆU (BÃO)
# ==============================================================================

def kt_to_bf(kt):
    if pd.isna(kt): return 0
    kt = float(kt)
    if kt < 1: return 0
    if kt < 34: return 6
    if kt < 64: return 8
    return 12

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
    # Logic đơn giản hóa để demo, bạn lắp logic chuẩn vào đây
    status = 'dubao' if 'forecast' in str(row.get('status_raw','')) else 'daqua'
    if w < 34: return f"vungthap_{status}"
    if w < 64: return f"atnd_{status}"
    if w < 100: return f"bnd_{status}"
    return f"sieubao_{status}"

# ==============================================================================
# 3. HTML DASHBOARD
# ==============================================================================

def create_info_table(df, title):
    """Bảng thông tin góc TRÊN PHẢI"""
    if df.empty:
        content = "<div style='padding:10px; text-align:center;'>Chưa có dữ liệu</div>"
    else:
        # Lấy tối đa 5 dòng mới nhất
        rows = ""
        for _, r in df.head(8).iterrows():
            t = r.get('datetime_str', r.get('dt'))
            if not isinstance(t, str): t = t.strftime('%d/%m %Hh')
            wind = int(r.get('wind_kt', 0))
            rows += f"<tr><td>{t}</td><td>{r['lat']:.1f}/{r['lon']:.1f}</td><td>{wind}</td></tr>"
        content = f"<table><thead><tr><th>Thời gian</th><th>Vị trí</th><th>Gió (kt)</th></tr></thead><tbody>{rows}</tbody></table>"

    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; top: 20px; right: 20px; width: 300px; background: white; border-radius: 8px; border: 1px solid #999; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="background:#007bff; color:white; padding:8px; text-align:center; font-weight:bold;">{title}</div>
        {content}
    </div>
    """)

def create_legend(img_b64):
    """Chú thích góc DƯỚI PHẢI"""
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; bottom: 30px; right: 20px; width: 260px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; border: 1px solid #999; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="text-align:center; font-weight:bold; font-size:12px; margin-bottom:5px;">CHÚ GIẢI</div>
        <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:4px;">
    </div>
    """)

# ==============================================================================
# 4. MAIN LOGIC (SƠ ĐỒ CÂY)
# ==============================================================================
def main():
    # Khởi tạo bản đồ
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    fg_storm = folium.FeatureGroup(name="Lớp Bão (Storm)")
    fg_weather = folium.FeatureGroup(name="Lớp Thời Tiết (Weather)")

    with st.sidebar:
        st.title("🎛️ ĐIỀU KHIỂN")
        
        # --- CẤP 1: CHỌN CHỦ ĐỀ ---
        topic = st.selectbox("1. CHỦ ĐỀ CHÍNH:", ["Bão (Typhoon)", "Thời tiết (Weather)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""

        # === NHÁNH 1: BÃO ===
        if topic == "Bão (Typhoon)":
            # --- CẤP 2: CHỨC NĂNG ---
            storm_opt = st.radio("2. CHỌN CHỨC NĂNG:", ["Option 1: Hiện trạng", "Option 2: Lịch sử"])
            st.markdown("---")
            
            if "Option 1" in storm_opt:
                dashboard_title = "TIN BÃO HIỆN TẠI"
                st.info("Đang ở Option 1: Xem hiện trạng")
                f = st.file_uploader("Upload besttrack.xlsx", type="xlsx", key="o1")
                path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                
                # ... Code xử lý Option 1 (như cũ) ...
                if path:
                    df = pd.read_excel(path)
                    # (Code xử lý load data option 1 giữ nguyên như các phiên bản trước)
                    # Ở đây tôi viết gọn để tập trung vào logic hạ tầng
                    rename_map = {"tên bão": "name", "biển đông": "storm_no", "vĩ độ": "lat", "kinh độ": "lon", "gió (kt)": "wind_kt", "Ngày - giờ": "datetime_str", "bán kính gió mạnh cấp 6 (km)": "r6", "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc"}
                    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
                    df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
                    df[['lat','lon','wind_kt']] = df[['lat','lon','wind_kt']].apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(subset=['lat','lon'])
                    
                    if 'storm_no' in df.columns:
                        all_s = df['storm_no'].unique()
                        sel = st.multiselect("Chọn bão:", all_s, default=all_s)
                        final_df = df[df['storm_no'].isin(sel)]
                    else: final_df = df
                    
                    # Vẽ Option 1
                    if not final_df.empty:
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
                                if os.path.exists(icon_path):
                                    folium.Marker([r['lat'],r['lon']], icon=folium.CustomIcon(icon_path, icon_size=(30,30))).add_to(fg_storm)
                                else:
                                    folium.CircleMarker([r['lat'],r['lon']], radius=3, color='black').add_to(fg_storm)

            else: # Option 2
                dashboard_title = "LỊCH SỬ BÃO"
                st.info("Đang ở Option 2: Xem lịch sử")
                f = st.file_uploader("Upload besttrack_capgio.xlsx", type="xlsx", key="o2")
                path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
                
                if path:
                    df = pd.read_excel(path)
                    # (Code load data option 2)
                    renames = {"tên bão":"name", "năm":"year", "tháng":"mon", "vĩ độ":"lat", "kinh độ":"lon", "gió (kt)":"wind_kt"}
                    df = df.rename(columns={k:v for k,v in renames.items() if k in df.columns})
                    # ... Xử lý ngày tháng ...
                    if all(c in df.columns for c in ['year','mon','day','hour']):
                        df['dt'] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day, hour=df.hour), errors='coerce')
                    
                    df = df.dropna(subset=['lat','lon','dt'])
                    
                    # BỘ LỌC CẤP 2 (Theo sơ đồ)
                    st.markdown("#### 🔍 Bộ Lọc")
                    years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                    temp = df[df['year'].isin(years)]
                    names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                    final_df = temp[temp['name'].isin(names)]
                    
                    # Vẽ Option 2
                    for n in final_df['name'].unique():
                        sub = final_df[final_df['name']==n].sort_values('dt')
                        folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg_storm)
                        for _, r in sub.iterrows():
                            folium.CircleMarker([r['lat'],r['lon']], radius=4, color='red', fill=True, popup=f"{n}").add_to(fg_storm)

        # === NHÁNH 2: THỜI TIẾT ===
        elif topic == "Thời tiết (Weather)":
            # --- CẤP 2: NGUỒN DỮ LIỆU ---
            weather_source = st.radio("2. NGUỒN DỮ LIỆU:", ["Option 3: Quan trắc", "Option 4: Mô hình"])
            st.markdown("---")
            
            # --- CẤP 3: THÔNG SỐ (Chung cho cả 2 nguồn) ---
            st.markdown("#### 3. Chọn Thông Số:")
            
            # Logic hiển thị theo sơ đồ:
            # Nếu Quan trắc -> Option 5, 6, 7
            # Nếu Mô hình -> Option 8, 9, 10
            
            w_param = st.radio("Thông số:", ["Nhiệt độ (Temp)", "Lượng mưa (Rain)", "Gió (Wind)"])
            
            st.success(f"Đang chọn: {weather_source} > {w_param}")
            
            # Logic giả lập (Mockup) để test hạ tầng
            if st.checkbox("Hiển thị lớp dữ liệu", value=True):
                if "Nhiệt độ" in w_param:
                    # Giả lập Heatmap
                    from folium.plugins import HeatMap
                    HeatMap([[16, 108, 30], [18, 110, 28]], radius=20).add_to(fg_weather)
                    dashboard_title = f"BẢN ĐỒ NHIỆT ĐỘ ({weather_source})"
                elif "Lượng mưa" in w_param:
                    # Giả lập vùng mưa
                    folium.Circle([15, 112], radius=50000, color='blue', fill=True).add_to(fg_weather)
                    dashboard_title = f"BẢN ĐỒ MƯA ({weather_source})"
                elif "Gió" in w_param:
                    # Giả lập hướng gió
                    folium.PolyLine([[10, 110], [12, 112]], color='green', arrow=True).add_to(fg_weather)
                    dashboard_title = f"BẢN ĐỒ GIÓ ({weather_source})"

    # --- RENDER GIAO DIỆN ---
    
    # 1. Vẽ các lớp
    fg_storm.add_to(m)
    fg_weather.add_to(m)
    
    # 2. Layer Control (Góc TRÁI TRÊN)
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)
    
    # 3. Dashboard (Góc PHẢI TRÊN)
    # Nếu đang chọn Bão và có dữ liệu -> Hiện bảng chi tiết
    if topic == "Bão (Typhoon)" and not final_df.empty:
        st.markdown(create_info_table(final_df, dashboard_title), unsafe_allow_html=True)
    # Nếu đang chọn Thời tiết -> Hiện bảng thông báo
    elif topic == "Thời tiết (Weather)":
        st.markdown(create_info_table(pd.DataFrame(), dashboard_title), unsafe_allow_html=True)
        
    # 4. Legend (Góc PHẢI DƯỚI - Chỉ hiện cho Option 1 Bão)
    if "Option 1" in str(active_mode := st.session_state.get('storm_opt', '')) and os.path.exists(CHUTHICH_IMG):
        with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        st.markdown(create_legend(b64), unsafe_allow_html=True)

    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
