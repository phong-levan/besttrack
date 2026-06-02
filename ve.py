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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
import geopandas as gpd
from shapely.geometry import Point, box, Polygon, mapping
from shapely.prepared import prep
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import zipfile
import shutil
import io
from datetime import datetime, timedelta
import branca.colormap as cm
import xarray as xr

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# --- CẤU HÌNH ĐƯỜNG DẪN SHAPEFILE CỐ ĐỊNH ---
SHP_MASK_PATH = os.path.join("shp", "vn34tinh.shp")
SHP_DISP_PATH = os.path.join("shp", "vungmoi.shp")

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

# --- HÀM TẠO LINK KMA DYNAMIC ---
def get_kma_url():
    now_utc = datetime.utcnow()
    check_time = now_utc - timedelta(hours=5)
    run_hour = 0 if check_time.hour < 12 else 12
    date_str = check_time.strftime("%Y.%m.%d")
    tm_str = f"{date_str}.{run_hour:02d}"
    url = f"https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm={tm_str}&delta=000&ftm={tm_str}"
    return url

COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
COLOR_TEXT = "#333333"
COLOR_ACCENT = "#007bff"
COLOR_BORDER = "#dee2e6"
SIDEBAR_WIDTH = "300px"

st.set_page_config(
    page_title="Hệ thống giám sát",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CSS CHUNG
# ==============================================================================
st.markdown(f"""
    <style>
    .block-container {{ padding: 0 !important; margin: 0 !important; max-width: 100% !important; }}
    header, footer {{ display: none !important; }}
    div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{
        visibility: hidden !important; display: none !important; height: 0px !important;
    }}
    section[data-testid="stSidebar"] {{
        display: block !important; visibility: visible !important;
        width: {SIDEBAR_WIDTH} !important; min-width: {SIDEBAR_WIDTH} !important; max-width: {SIDEBAR_WIDTH} !important;
        position: fixed !important; left: 0 !important; top: 0 !important; height: 100vh !important;
        transform: none !important; z-index: 100000 !important; background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd;
    }}
    [data-testid="stSidebarCollapseBtn"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; padding-top: 0 !important; }}
    [data-testid="stMainViewContainer"] {{ margin-left: 0 !important; width: 100% !important; padding-top: 0 !important; }}
    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; display: block !important; }}
    .floating-container {{ position: fixed; top: 20px; right: 60px; z-index: 9999; display: flex; flex-direction: column; align-items: center; }}
    .legend-box {{ width: 300px; pointer-events: none; margin-bottom: 5px; }}
    .info-box {{
        width: fit-content; background: rgba(255, 255, 255, 0.9); border: 1px solid #ccc; border-radius: 6px;
        padding: 5px !important; color: #000; text-align: center;
    }}
    .info-box table {{ width: 100%; margin: 0 auto; border-collapse: collapse; }}
    .info-box th, .info-box td {{ text-align: center !important; padding: 2px 5px !important; font-size: 12px !important; }}
    .info-title {{ font-weight: bold; margin-bottom: 2px; font-size: 14px !important; }}
    .info-subtitle {{ font-size: 10px !important; margin-bottom: 5px; font-style: italic; }}
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
    if not os.path.exists(image_path): return None
    with open(image_path, "rb") as f: encoded = base64.b64encode(f.read()).decode()
    ext = image_path.split('.')[-1].lower()
    mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {
        "tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no", "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "thời gian (giờ)": "hour_explicit", "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
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
    status = 'daqua' if 'quá khứ' in str(row.get('status_raw','')).lower() or 'past' in str(row.get('status_raw','')).lower() else 'dubao'
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
        cur = display_df 

    subtitle = "(Đang cập nhật)"
    try:
        target_row = cur.iloc[0] if not cur.empty else (display_df.iloc[0] if not display_df.empty else None)
        if target_row is not None:
            if 'hour_explicit' in target_row and pd.notna(target_row['hour_explicit']): subtitle = f"Tin phát lúc {int(target_row['hour_explicit'])}h30"
            elif 'dt' in target_row and pd.notna(target_row['dt']): subtitle = f"Tin phát lúc {target_row['dt'].hour}h30"
    except: subtitle = "(Dữ liệu cập nhật từ Besttrack)"
    
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
    return textwrap.dedent(f"""<div class="info-box"><div class="info-title">{title}</div><div class="info-subtitle">{subtitle}</div><table><thead><tr><th>Ngày-Giờ</th><th>Kinh độ</th><th>Vĩ độ</th><th>Cấp gió</th><th>Pmin (hPa)</th></tr></thead><tbody>{rows}</tbody></table></div>""")

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

def run_interpolation_and_plot(input_df, title_text, data_type='temp'):
    minx, maxx, miny, maxy = 101.8, 115.0, 8.0, 23.9
    GRID_N, SIGMA, IDW_POWER, KNN = 1000, 1.5, 3.0, 12

    if data_type == 'rain':
        vmin, vmax = 0, 1400
        levels_for_ticks = np.arange(0, 1450, 100)
        colors = ['#FFFFFF', '#A0E6FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800080', '#4B0082']
        cmap = LinearSegmentedColormap.from_list('rain_smooth', colors, N=512)
        cmap.set_under(colors[0]); cmap.set_over(colors[-1])
        unit_label = "Lượng mưa (mm)"
    else: 
        vmin, vmax = 0.0, 40.0
        levels_for_ticks = list(range(0, 42, 4))
        colors = [(0.0, '#FFFFFF'), (0.1, '#D0F0FF'), (0.2, '#00A0FF'), (0.4, '#00FF00'), (0.6, '#FFFF00'), (0.75, '#FFA500'), (0.9, '#FF0000'), (1.0, '#8B0000')]
        cmap = LinearSegmentedColormap.from_list("custom_smooth_temp", colors, N=256)
        unit_label = "Nhiệt độ (°C)"

    norm = Normalize(vmin=vmin, vmax=vmax)
    input_df.columns = input_df.columns.str.lower().str.strip()
    if not all(c in input_df.columns for c in ['lon', 'lat', 'value']): return None, "File thiếu cột bắt buộc."
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, "Dữ liệu trống."

    x_pts, y_pts, z_pts = valid['lon'].to_numpy(), valid['lat'].to_numpy(), valid['value'].to_numpy()
    edge_points = pd.DataFrame({'lon': [minx, minx, maxx, maxx, (minx + maxx)/2], 'lat': [miny, maxy, miny, maxy, (miny + maxy)/2], 'value': [float(np.nanmean(z_pts))] * 5})
    aug = pd.concat([valid[['lon', 'lat', 'value']], edge_points], ignore_index=True)
    xi, yi, zi = aug['lon'].to_numpy(), aug['lat'].to_numpy(), aug['value'].to_numpy()

    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(xi, yi, zi, grid_xy, k=KNN, power=IDW_POWER).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    mask_shape = disp_shape = None
    if os.path.exists(SHP_MASK_PATH):
        try:
            mask_shape = gpd.read_file(SHP_MASK_PATH)
            if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)
        except Exception as e: return None, f"Lỗi đọc Mask Shapefile: {e}"
    else:
        mask_shape = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

    if os.path.exists(SHP_DISP_PATH):
        try:
            disp_shape = gpd.read_file(SHP_DISP_PATH)
            if disp_shape.crs and disp_shape.crs.to_epsg() != 4326: disp_shape.to_crs(
