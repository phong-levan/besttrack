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
import io
from datetime import datetime, timedelta
import branca.colormap as cm

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CẤU HÌNH FONT TIẾNG VIỆT
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")
SHP_MASK_PATH = os.path.join("shp", "vn34tinh.shp")
SHP_DISP_PATH = os.path.join("shp", "vungmoi.shp")

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

LINK_WEATHEROBS = "https://weatherobs.com/"
LINK_WIND_AUTO = "https://kttvtudong.net/kttv"

def get_kma_url():
    now_utc = datetime.utcnow()
    check_time = now_utc - timedelta(hours=5)
    run_hour = 0 if check_time.hour < 12 else 12
    date_str = check_time.strftime("%Y.%m.%d")
    tm_str = f"{date_str}.{run_hour:02d}"
    return f"https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm={tm_str}&delta=000&ftm={tm_str}"

COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
SIDEBAR_WIDTH = "300px"

st.set_page_config(page_title="Hệ thống giám sát", layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown(f"""
    <style>
    .block-container {{ padding: 0 !important; margin: 0 !important; max-width: 100% !important; }}
    header, footer {{ display: none !important; }}
    div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{ visibility: hidden !important; display: none !important; }}
    section[data-testid="stSidebar"] {{ width: {SIDEBAR_WIDTH} !important; background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd; }}
    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; }}
    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; }}
    .floating-container {{ position: fixed; top: 20px; right: 60px; z-index: 9999; display: flex; flex-direction: column; align-items: center; }}
    .info-box {{ background: rgba(255, 255, 255, 0.9); border: 1px solid #ccc; border-radius: 6px; padding: 5px; color: #000; text-align: center; }}
    .info-title {{ font-weight: bold; font-size: 14px; }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HÀM XỬ LÝ LOGIC
# ==============================================================================

@st.cache_data(ttl=300)
def get_rainviewer_ts():
    try:
        r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=3, verify=False)
        return r.json()['satellite']['infrared'][-1]['time']
    except: return None

def image_to_base64(image_path):
    if not os.path.exists(image_path): return None
    with open(image_path, "rb") as f: return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {"tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no", "ngày - giờ": "datetime_str", "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h", "cường độ (cấp bf)": "bf"}
    return df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

def densify_track(df, step_km=10):
    if len(df) < 2: return df
    new_rows = []
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = 6371 * 2 * asin(sqrt(sin(radians(p2['lat']-p1['lat'])/2)**2 + cos(radians(p1['lat']))*cos(radians(p2['lat']))*sin(radians(p2['lon']-p1['lon'])/2)**2))
        steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(steps):
            f = j / steps
            row = p1.copy()
            row['lat'] = p1['lat'] + (p2['lat'] - p1['lat']) * f
            row['lon'] = p1['lon'] + (p2['lon'] - p1['lon']) * f
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
    return u['r6'], u['r10'], u['rc']

def get_icon_name(row):
    bf = row.get('bf', 0)
    status = 'daqua' if 'quá khứ' in str(row.get('status_raw','')).lower() else 'dubao'
    if bf < 6: return f"vungthap_{status}"
    if bf < 8: return f"atnd_{status}"
    if bf <= 11: return f"bnd_{status}"
    return f"sieubao_{status}"

def idw_knn(xi, yi, zi, query_xy, k=12, power=3.0):
    tree = cKDTree(np.column_stack([xi, yi]))
    dists, idxs = tree.query(query_xy, k=min(k, xi.size))
    if dists.ndim == 1: dists, idxs = dists[:, None], idxs[:, None]
    w = 1.0 / np.maximum(dists, 1e-12)**power
    return (w * zi[idxs]).sum(axis=1) / w.sum(axis=1)

def run_interpolation_and_plot(input_df, title_text, data_type='temp'):
    minx, maxx, miny, maxy = 101.8, 115.0, 8.0, 23.9
    valid = input_df.dropna(subset=['lon', 'lat', 'value'])
    gx, gy = np.meshgrid(np.linspace(minx, maxx, 800), np.linspace(miny, maxy, 800))
    gv = idw_knn(valid['lon'].values, valid['lat'].values, valid['value'].values, np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    gv = gaussian_filter(gv, sigma=1.5)
    
    mask_shape = gpd.read_file(SHP_MASK_PATH) if os.path.exists(SHP_MASK_PATH) else None
    if mask_shape is not None:
        prep_s = prep(mask_shape.unary_union)
        mask_flat = np.array([prep_s.contains(Point(x, y)) for x, y in np.column_stack([gx.ravel(), gy.ravel()])]).reshape(gx.shape)
        gv = np.where(mask_flat, gv, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(gv, extent=[minx, maxx, miny, maxy], origin='lower', cmap='jet')
    plt.colorbar(im, ax=ax)
    if mask_shape is not None: mask_shape.boundary.plot(ax=ax, color='black', linewidth=0.5)
    return fig, None

# ==============================================================================
# HÀM NỘI SUY LINH TINH (BỎ NỀN - FIX CỨNG PHẠM VI)
# ==============================================================================
def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None):
    valid = input_df.dropna(subset=['lon', 'lat', 'value'])
    mask_shape = gpd.read_file(SHP_MASK_PATH)
    
    # Lọc tỉnh
    if selected_provinces:
        display_shape = mask_shape[mask_shape[shape_col].isin(selected_provinces)]
    else:
        display_shape = mask_shape

    minx, miny, maxx, maxy = display_shape.total_bounds
    gx, gy = np.meshgrid(np.linspace(minx, maxx, 800), np.linspace(miny, maxy, 800))
    gv = idw_knn(valid['lon'].values, valid['lat'].values, valid['value'].values, np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    
    prep_s = prep(display_shape.unary_union)
    mask_flat = np.array([prep_s.contains(Point(x, y)) for x, y in np.column_stack([gx.ravel(), gy.ravel()])]).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    cmap = plt.get_cmap(cmap_name)
    if custom_levels is None: custom_levels = np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), num_bins + 1)
    norm = BoundaryNorm(custom_levels, ncolors=cmap.N)

    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]
    
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    img_base64 = base64.b64encode(buf.read()).decode()

    # KHỞI TẠO BẢN ĐỒ KHÔNG NỀN (tiles=None)
    m = folium.Map(
        location=[(miny + maxy) / 2, (minx + maxx) / 2], 
        zoom_start=6, 
        tiles=None,
        max_bounds=True,
        min_lat=miny-2, max_lat=maxy+2, min_lon=minx-2, max_lon=maxx+2
    )
    
    # Lớp nền trắng (Shapefile 34 tỉnh)
    folium.GeoJson(
        mask_shape,
        style_function=lambda x: {'fillColor': '#ffffff', 'color': '#cccccc', 'weight': 0.5, 'fillOpacity': 1}
    ).add_to(m)

    # Lớp ảnh nội suy
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_base64}",
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=0.9
    ).add_to(m)

    # Viền tỉnh phía trên
    folium.GeoJson(
        display_shape,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1}
    ).add_to(m)

    colormap_branca = cm.StepColormap(
        colors=[mcolors.to_hex(cmap(norm(val))) for val in custom_levels[:-1]],
        vmin=custom_levels[0], vmax=custom_levels[-1], index=custom_levels, caption=title_text
    )
    m.add_child(colormap_branca)

    # Tạo Figure tĩnh
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], origin='lower', cmap=cmap, norm=norm, alpha=0.9)
    display_shape.boundary.plot(ax=ax, color='black', linewidth=0.8)
    ax.set_facecolor('white')
    ax.set_title(title_text)

    return m, fig, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    for key in ['interpol_fig', 'folium_map_obj', 'folium_fig_obj', 'logged_in']:
        if key not in st.session_state: st.session_state[key] = False

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        
        final_df = pd.DataFrame()
        show_widgets = False
        
        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Chọn nguồn:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy nhiệt độ", "Nội suy lượng mưa", "Nội suy linh tinh"])
            
            if "Nội suy" in obs_mode:
                title_interpol = st.text_input("Tiêu đề:", value="Bản đồ Nội suy")
                data_file_interpol = st.file_uploader("Chọn file:", type=['xlsx', 'csv'])
                
                if obs_mode == "Nội suy linh tinh":
                    cmap_option = st.selectbox("Màu:", plt.colormaps(), index=plt.colormaps().index('jet'))
                    num_bins = st.number_input("Số lớp:", 2, 50, 10)
                    selected_provinces = st.multiselect("Chọn tỉnh:", []) # Cần logic đọc tỉnh từ SHP nếu muốn
                    btn_run = st.button("🚀 VẼ TƯƠNG TÁC")
                else:
                    btn_run = st.button("🚀 VẼ TĨNH")

    # Xử lý Main Content (Rút gọn logic hiển thị)
    if topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            with st.form("login"):
                u, p = st.text_input("User"), st.text_input("Pass", type="password")
                if st.form_submit_button("Login") and u == "admin" and p == "kttv@2026":
                    st.session_state['logged_in'] = True
                    st.rerun()
        else:
            if obs_mode == "Nội suy linh tinh" and btn_run and data_file_interpol:
                df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)
                m, fig, err = run_interactive_folium_interpolation(df_in, title_interpol, cmap_option, num_bins, None, [], "NAME_1")
                st.session_state['folium_map_obj'], st.session_state['folium_fig_obj'] = m, fig
            
            if st.session_state['folium_map_obj'] and obs_mode == "Nội suy linh tinh":
                st_folium(st.session_state['folium_map_obj'], width=None, height=800, use_container_width=True, returned_objects=[])

    # Các phần khác (Bão, Vệ tinh...) giữ nguyên logic như code cũ của bạn
    elif topic == "Bản đồ Bão":
        st.info("Chức năng Bản đồ Bão hiển thị tại đây...")

if __name__ == "__main__":
    main()
