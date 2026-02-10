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
import zipfile
import tempfile
import shutil
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, box, Polygon, mapping
from shapely.prepared import prep
from shapely.ops import unary_union
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

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
    page_title="Hệ thống giám sát",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CSS CHUNG (FIX CỨNG SIDEBAR, ĐẨY NỘI DUNG & XÓA KHOẢNG TRẮNG)
# ==============================================================================
st.markdown(f"""
    <style>
    /* 1. XÓA SẠCH PADDING/MARGIN ĐỂ FULL MÀN HÌNH */
    .block-container {{
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }}
    
    header, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }}

    /* 2. ÉP SIDEBAR CỐ ĐỊNH */
    section[data-testid="stSidebar"] {{
        display: block !important;
        width: {SIDEBAR_WIDTH} !important;
        min-width: {SIDEBAR_WIDTH} !important;
        max-width: {SIDEBAR_WIDTH} !important;
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        height: 100vh !important;
        z-index: 100000 !important;
        background-color: {COLOR_SIDEBAR} !important;
        border-right: 1px solid #ddd;
    }}

    /* 3. ĐẨY NỘI DUNG CHÍNH */
    [data-testid="stAppViewContainer"] {{
        padding-left: {SIDEBAR_WIDTH} !important;
        padding-top: 0 !important;
    }}
    [data-testid="stMainViewContainer"] {{
        margin-left: 0 !important;
        width: 100% !important;
        padding-top: 0 !important;
    }}

    iframe {{
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
    }}

    .floating-container {{
        position: fixed; 
        top: 20px; 
        right: 60px; 
        z-index: 9999;
        display: flex;
        flex-direction: column; 
        align-items: center;    
    }}

    .legend-box {{
        width: 340px; 
        pointer-events: none;
        margin-bottom: 5px; 
    }}
    
    .info-box {{
        width: fit-content; 
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #ccc; 
        border-radius: 6px;
        padding: 10px !important; 
        color: #000;
        text-align: center;
    }}
    
    .info-box table {{ width: 100%; margin: 0 auto; border-collapse: collapse; }}
    .info-box th, .info-box td {{ text-align: center !important; padding: 4px 8px; }}
    .info-title {{ font-weight: bold; margin-bottom: 2px; }}
    .info-subtitle {{ font-size: 0.9em; margin-bottom: 8px; font-style: italic; }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HÀM XỬ LÝ LOGIC CHUNG
# ==============================================================================

@st.cache_data(ttl=300) 
def get_rainviewer_ts():
    try:
        url = "https://api.rainviewer.com/public/weather-maps.json"
        r = requests.get(url, timeout=3, verify=False)
        return r.json()['satellite']['infrared'][-1]['time']
    except: return None

def image_to_base64(image_path):
    if not os.path.exists(image_path): return None
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
        "thời gian (giờ)": "hour_explicit", 
        "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "khí áp": "pressure", "khí áp (mb)": "pressure", "pmin": "pressure", "pmin (mb)": "pressure"
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    return df

# --- Logic Bão ---
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
    if radius_km <= 0: return None
    coords = []
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
    return (u['r6'].difference(u['r10']) if u['r6'] and u['r10'] else u['r6']), \
           (u['r10'].difference(u['rc']) if u['r10'] and u['rc'] else u['r10']), \
           u['rc']

def get_icon_name(row):
    wind_speed = row.get('bf', 0) 
    w = row.get('wind_km/h', 0)
    if pd.isna(wind_speed) or wind_speed == 0:
        if w > 0:
            if w < 34: wind_speed = 5
            elif w < 64: wind_speed = 7
            elif w < 100: wind_speed = 10
            else: wind_speed = 12
    status = 'daqua' if 'quá khứ' in str(row.get('status_raw','')).lower() or 'past' in str(row.get('status_raw','')).lower() else 'dubao'
    if pd.isna(wind_speed): return f"vungthap_{status}"
    if wind_speed < 6: return f"vungthap_{status}"
    if wind_speed < 8: return f"atnd_{status}"
    if wind_speed <= 11: return f"bnd_{status}"
    return f"sieubao_{status}"

def create_info_table(df, title):
    if df.empty: return ""
    if 'status_raw' in df.columns:
        cur = df[df['status_raw'].astype(str).str.contains("hiện tại|current", case=False, na=False)]
        fut = df[df['status_raw'].astype(str).str.contains("dự báo|forecast", case=False, na=False)]
        display_df = pd.concat([cur, fut]).head(8)
    else:
        display_df = df.sort_values('dt', ascending=False).groupby('name').head(1)
        cur = display_df 

    subtitle = "(Đang cập nhật)"
    try:
        target = cur.iloc[0] if not cur.empty else (display_df.iloc[0] if not display_df.empty else None)
        if target is not None:
            if 'hour_explicit' in target and pd.notna(target['hour_explicit']): subtitle = f"Tin phát lúc {int(target['hour_explicit'])}h30"
            elif 'dt' in target and pd.notna(target['dt']): subtitle = f"Tin phát lúc {target['dt'].hour}h30"
    except: pass
    
    rows = ""
    for _, r in display_df.iterrows():
        t = r.get('datetime_str', r.get('dt'))
        if not isinstance(t, str): t = t.strftime('%d/%m %Hh')
        w = r.get('wind_km/h', 0)
        bf = r.get('bf', 0)
        if (pd.isna(bf) or bf == 0) and w > 0:
             if w < 34: bf = 6
             elif w < 64: bf = 8
             elif w < 100: bf = 10
             else: bf = 12
        rows += f"<tr><td>{t}</td><td>{r.get('lon',0):.1f}E</td><td>{r.get('lat',0):.1f}N</td><td>{f'Cấp {int(bf)}' if bf>0 else '-'}</td><td>{f'{int(r.get('pressure',0))}' if r.get('pressure',0)>0 else '-'}</td></tr>"
    
    return f'<div class="info-box"><div class="info-title">{title}</div><div class="info-subtitle">{subtitle}</div><table><thead><tr><th>Ngày-Giờ</th><th>Kinh độ</th><th>Vĩ độ</th><th>Cấp bão</th><th>Pmin</th></tr></thead><tbody>{rows}</tbody></table></div>'

def create_legend(img_b64):
    return f'<div class="legend-box"><img src="data:image/png;base64,{img_b64}"></div>' if img_b64 else ""

# ==============================================================================
# 4. LOGIC NỘI SUY (MỚI)
# ==============================================================================
def idw_knn(xi, yi, zi, query_xy, k=12, power=3.0, eps=1e-12):
    tree = cKDTree(np.column_stack([xi, yi]))
    dists, idxs = tree.query(query_xy, k=min(k, xi.size))
    if dists.ndim == 1: dists, idxs = dists[:, None], idxs[:, None]
    
    exact = dists <= eps
    out = np.empty(dists.shape[0], dtype=float)
    if np.any(exact):
        for r in np.where(exact.any(axis=1))[0]:
            out[r] = zi[idxs[r, np.where(exact[r])[0][0]]]
            
    rest = ~exact.any(axis=1)
    if np.any(rest):
        d, nn = dists[rest], idxs[rest]
        w = 1.0 / np.maximum(d, eps)**power
        out[rest] = (w * zi[nn]).sum(axis=1) / w.sum(axis=1)
    return out

def run_interpolation(data, shape_gdf=None):
    # Cấu hình
    minx, maxx = 101.8, 115.0
    miny, maxy = 8.0, 23.9
    GRID_N = 1000  # Giảm xuống 1000 cho web chạy nhanh hơn
    SIGMA = 1.5
    KNN = 12
    IDW_POWER = 3.0

    valid = data.dropna(subset=['lon', 'lat', 'value']).copy()
    xi, yi, zi = valid['lon'].to_numpy(), valid['lat'].to_numpy(), valid['value'].to_numpy()

    # Thêm điểm biên
    edge_val = float(np.nanmean(zi))
    aug_xi = np.append(xi, [minx, minx, maxx, maxx, (minx+maxx)/2])
    aug_yi = np.append(yi, [miny, maxy, miny, maxy, (miny+maxy)/2])
    aug_zi = np.append(zi, [edge_val]*5)

    # Tạo lưới
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    # IDW
    gv = idw_knn(aug_xi, aug_yi, aug_zi, grid_xy, k=KNN, power=IDW_POWER).reshape(gx.shape)
    
    # Gaussian Smooth
    gv = gaussian_filter(gv, sigma=SIGMA)

    # Masking
    if shape_gdf is not None:
        try:
            # Tạo mask
            mask_poly = shape_gdf.unary_union
            prep_shape = prep(mask_poly)
            # Tối ưu check points
            mask_flat = np.array([prep_shape.contains(Point(px, py)) for px, py in grid_xy]).reshape(gx.shape)
            gv_masked = np.where(mask_flat, gv, np.nan)
        except:
            gv_masked = gv # Fallback nếu lỗi mask
    else:
        gv_masked = gv

    # Vẽ
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Màu sắc
    colors = [(0.0, '#FFFFFF'), (0.1, '#D0F0FF'), (0.2, '#00A0FF'), (0.4, '#00FF00'), 
              (0.6, '#FFFF00'), (0.75, '#FFA500'), (0.9, '#FF0000'), (1.0, '#8B0000')]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
    norm = Normalize(vmin=0, vmax=40)

    # Plot shape boundary
    if shape_gdf is not None:
        shape_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
    else:
        # Vẽ khung VN sơ bộ nếu ko có shape
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, interpolation='bilinear', origin='lower')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Nhiệt độ (°C)')
    
    ax.set_title(f"Bản đồ nội suy nhiệt độ (IDW + Gaussian)", fontsize=14)
    return fig

# ==============================================================================
# 5. MAIN APP
# ==============================================================================
def main():
    
    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""
        obs_mode = ""

        if topic == "Dữ liệu quan trắc":
            obs_mode = st.radio("Chọn nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy nhiệt độ"])

        if topic == "Bản đồ Bão":
            storm_opt = st.selectbox("Dữ liệu bão:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])
            active_mode = storm_opt
            if "Hiện trạng" in storm_opt:
                dashboard_title = "TIN BÃO KHẨN CẤP"
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack.csv", type="csv", key="o1")
                    path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                    if path:
                        try:
                            df = pd.read_csv(path) if (isinstance(path, str) and path.endswith('.csv')) or (not isinstance(path, str) and path.name.endswith('.csv')) else pd.read_excel(path)
                            df = normalize_columns(df)
                            if 'name' not in df: df['name'], df['storm_no'] = 'Storm', 'Current'
                            for c in ['wind_km/h','bf','r6','r10','rc','pressure','hour_explicit']: 
                                if c not in df: df[c]=0
                            df = df.dropna(subset=['lat','lon'])
                            all_s = df['storm_no'].unique() if 'storm_no' in df else []
                            sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s) if len(all_s)>0 else []
                            final_df = df[df['storm_no'].isin(sel)] if len(sel)>0 else df
                        except: pass
            else:
                dashboard_title = "THỐNG KÊ LỊCH SỬ"
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack_capgio.xlsx", type="xlsx", key="o2")
                    path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
                    if path:
                        try:
                            df = pd.read_excel(path)
                            df = normalize_columns(df)
                            df = df.dropna(subset=['lat','lon'])
                            years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                            temp = df[df['year'].isin(years)]
                            names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                            final_df = temp[temp['name'].isin(names)]
                        except: pass

    # --- MAIN DISPLAY ---
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=1000&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1")
    
    elif topic == "Dữ liệu quan trắc":
        if "WeatherObs" in obs_mode:
            st.markdown(f'<div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;"><iframe src="{LINK_WEATHEROBS}" style="width: calc(100% + 19px); height: 1000px; position: absolute; top: -65px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
        elif "Gió tự động" in obs_mode:
            st.markdown(f'<div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;"><iframe src="{LINK_WIND_AUTO}" style="width: calc(100% + 19px); height: 1200px; position: absolute; top: -100px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
        elif "Nội suy nhiệt độ" in obs_mode:
            st.markdown("### 🌡️ Nội suy Nhiệt độ từ trạm Quan trắc")
            col1, col2 = st.columns(2)
            with col1:
                data_file = st.file_uploader("1. Upload dữ liệu (Excel/CSV) - Cột: stations, lon, lat, value", type=["xlsx", "xls", "csv"])
            with col2:
                shape_file = st.file_uploader("2. Upload Shapefile biên giới (ZIP) - Tùy chọn", type=["zip"])
            
            if data_file:
                if st.button("🚀 Thực hiện Nội suy"):
                    with st.spinner("Đang xử lý dữ liệu và vẽ bản đồ..."):
                        try:
                            # Đọc Data
                            df_in = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
                            req = ['lon', 'lat', 'value']
                            if not all(c in df_in.columns for c in req):
                                st.error(f"File thiếu cột bắt buộc: {req}")
                            else:
                                # Đọc Shapefile nếu có
                                gdf_shape = None
                                if shape_file:
                                    with tempfile.TemporaryDirectory() as tmpdir:
                                        with zipfile.ZipFile(shape_file, 'r') as z: z.extractall(tmpdir)
                                        shps = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                                        if shps:
                                            gdf_shape = gpd.read_file(os.path.join(tmpdir, shps[0]))
                                            if gdf_shape.crs and gdf_shape.crs.to_epsg() != 4326:
                                                gdf_shape = gdf_shape.to_crs(epsg=4326)
                                
                                # Chạy nội suy
                                fig = run_interpolation(df_in, gdf_shape)
                                st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Có lỗi xảy ra: {e}")

    elif topic == "Dự báo điểm (KMA)":
        st.markdown(f'<div style="overflow: hidden; width: 100%; height: 700px; position: relative; border: 1px solid #ddd;"><iframe src="{LINK_KMA_FORECAST}" style="width: calc(100% + 19px); height: 1200px; position: absolute; top: -215px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)

    elif topic == "Bản đồ Bão":
        m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
        folium.TileLayer('CartoDB positron', name='Bản đồ Sáng', overlay=False, control=True).add_to(m)
        folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết', overlay=False, control=True).add_to(m)
        folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Vệ tinh', overlay=False, control=True).add_to(m)
        
        ts = get_rainviewer_ts()
        if ts: folium.TileLayer(tiles=f"https://tile.rainviewer.com/{ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png", attr="RainViewer", name="☁️ Mây Vệ tinh", overlay=True, show=True, opacity=0.5).add_to(m)

        fg = folium.FeatureGroup(name="🌀 Đường đi Bão")
        if not final_df.empty and show_widgets:
            if "Hiện trạng" in str(active_mode):
                groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
                for g in groups:
                    sub = final_df[final_df['storm_no']==g] if g else final_df
                    dense = densify_track(sub)
                    f6, f10, fc = create_storm_swaths(dense)
                    for geom, c, o in [(f6,'#FFC0CB',0.4), (f10,'#FF6347',0.5), (fc,'#90EE90',0.6)]:
                         if geom and not geom.is_empty: folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg)
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg)
                    for _, r in sub.iterrows():
                        icon = folium.CustomIcon(image_to_base64(ICON_PATHS.get(get_icon_name(r))), icon_size=(40,40) if 'vungthap' not in get_icon_name(r) else (20,20)) if image_to_base64(ICON_PATHS.get(get_icon_name(r))) else None
                        if icon: folium.Marker([r['lat'], r['lon']], icon=icon, tooltip=f"Gió: {r.get('wind_km/h',0)} km/h").add_to(fg)
            else:
                for n in final_df['name'].unique():
                    sub = final_df[final_df['name']==n].sort_values('dt')
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg)
                    for _, r in sub.iterrows():
                        folium.CircleMarker([r['lat'],r['lon']], radius=3, color='#00f2ff' if r.get('wind_km/h',0)<64 else '#ff0055', fill=True, popup=n).add_to(fg)
        
        fg.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        
        if show_widgets:
            html = '<div class="floating-container">'
            if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG):
                with open(CHUTHICH_IMG, "rb") as f: html += create_legend(base64.b64encode(f.read()).decode())
            html += create_info_table(final_df if not final_df.empty else pd.DataFrame(), dashboard_title if not final_df.empty else "ĐANG TẢI DỮ LIỆU...")
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
        
        st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
