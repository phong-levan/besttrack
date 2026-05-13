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
SHP_MASK_PATH = os.path.join("shp", "vn34tinh.shp")
SHP_DISP_PATH = os.path.join("shp", "vungmoi.shp")
QN_GDB_PATH = "nen.gdb" # Đường dẫn File Geodatabase cho Quảng Ninh

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
    url = f"https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm={tm_str}&delta=000&ftm={tm_str}"
    return url

COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
COLOR_TEXT = "#333333"
COLOR_ACCENT = "#007bff"
COLOR_BORDER = "#dee2e6"
SIDEBAR_WIDTH = "300px"

st.set_page_config(page_title="Hệ thống giám sát", layout="wide", initial_sidebar_state="expanded")

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
    dist_res, idx_res = tree.query(query_xy, k=min(k, xi.size))
    if dist_res.ndim == 1: dist_res, idx_res = dist_res[:, None], idx_res[:, None]
    exact = dist_res <= eps
    out = np.empty(dist_res.shape[0], dtype=float)
    if np.any(exact):
        for r in np.where(exact.any(axis=1))[0]:
            out[r] = zi[idx_res[r, np.where(exact[r])[0][0]]]
    rest = ~exact.any(axis=1)
    if np.any(rest):
        d, nn = dist_res[rest], idx_res[rest]
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
        unit_label = "Lượng mưa (mm)"
    else: 
        vmin, vmax = 0.0, 40.0
        levels_for_ticks = list(range(0, 42, 4))
        colors = [(0.0, '#FFFFFF'), (0.1, '#D0F0FF'), (0.2, '#00A0FF'), (0.4, '#00FF00'), (0.6, '#FFFF00'), (0.75, '#FFA500'), (0.9, '#FF0000'), (1.0, '#8B0000')]
        cmap = LinearSegmentedColormap.from_list("custom_smooth_temp", colors, N=256)
        unit_label = "Nhiệt độ (°C)"
    norm = Normalize(vmin=vmin, vmax=vmax)
    input_df.columns = input_df.columns.str.lower().str.strip()
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, "Dữ liệu trống."
    xi, yi, zi = valid['lon'].to_numpy(), valid['lat'].to_numpy(), valid['value'].to_numpy()
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(xi, yi, zi, grid_xy, k=KNN, power=IDW_POWER).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)
    mask_shape = gpd.read_file(SHP_MASK_PATH) if os.path.exists(SHP_MASK_PATH) else None
    if mask_shape is not None:
        prep_shape = prep(mask_shape.unary_union)
        mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
        gv_masked = np.where(mask_flat, gv, np.nan)
    else: gv_masked = gv
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, interpolation='bilinear', origin='lower')
    plt.colorbar(im, ax=ax, shrink=0.7, label=unit_label)
    return fig, None

def generate_single_province_fig(cache, prov_name, title_text):
    mask_shape = cache.get('mask_shape')
    shape_col = cache.get('shape_col', "")
    if mask_shape is None or not shape_col: return None
    prov_shape = mask_shape[mask_shape[shape_col] == prov_name]
    if prov_shape.empty: return None
    p_minx, p_miny, p_maxx, p_maxy = prov_shape.total_bounds
    grid_xy = np.column_stack([cache['gx'].ravel(), cache['gy'].ravel()])
    prep_shape = prep(prov_shape.unary_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(cache['gx'].shape)
    gv_masked = np.where(mask_flat, cache['gv'], np.nan)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{title_text}\n(Khu vực: {prov_name})")
    ax.imshow(gv_masked, extent=[cache['minx'], cache['maxx'], cache['miny'], cache['maxy']], cmap=cache['cmap'], norm=cache['norm'], origin='lower')
    ax.set_xlim(p_minx, p_maxx); ax.set_ylim(p_miny, p_maxy)
    return fig

def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None, mask_gdf=None):
    input_df.columns = input_df.columns.str.lower().str.strip()
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, None, None, "Dữ liệu trống."

    # Xử lý ranh giới Mask
    if mask_gdf is not None:
        mask_shape = mask_gdf
    elif os.path.exists(SHP_MASK_PATH):
        mask_shape = gpd.read_file(SHP_MASK_PATH)
        if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)
    else: return None, None, None, "Thiếu file ranh giới."

    if selected_provinces and shape_col and shape_col in mask_shape.columns:
        mask_shape = mask_shape[mask_shape[shape_col].isin(selected_provinces)]

    if custom_bounds:
        minx, maxx, miny, maxy = custom_bounds['minx'], custom_bounds['maxx'], custom_bounds['miny'], custom_bounds['maxy']
        mask_shape = gpd.clip(mask_shape, box(minx, miny, maxx, maxy))
    else:
        minx, miny, maxx, maxy = mask_shape.total_bounds
    
    # Chỉ lấy dữ liệu điểm nằm TRONG ranh giới mask
    pts_geom = [Point(x, y) for x, y in zip(valid['lon'], valid['lat'])]
    valid_gdf = gpd.GeoDataFrame(valid, geometry=pts_geom, crs="EPSG:4326")
    valid = gpd.sjoin(valid_gdf, mask_shape, how="inner").drop(columns='index_right')
    if valid.empty: return None, None, None, "Không có dữ liệu điểm nào nằm trong ranh giới vùng đã chọn."

    GRID_N, SIGMA = 800, 1.0
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(valid['lon'].values, valid['lat'].values, valid['value'].values, grid_xy, k=12, power=3.0).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    shape_union = mask_shape.unary_union
    prep_shape = prep(shape_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    cmap = plt.get_cmap(cmap_name)
    if custom_levels:
        custom_levels = sorted(list(set(custom_levels)))
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')
    else:
        v_min, v_max = np.nanmin(gv_masked), np.nanmax(gv_masked)
        custom_levels = np.linspace(v_min if not np.isnan(v_min) else 0, v_max if not np.isnan(v_max) else 1, num_bins + 1)
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')

    # Create Folium Map
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]
    img_base64 = base64.b64encode(io.BytesIO(plt.imsave(io.BytesIO(), np.flipud(rgba), format='png')).getvalue()).decode()
    
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], tiles="CartoDB positron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_base64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.75).add_to(m)
    folium.GeoJson(mask_shape, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1.0}).add_to(m)
    
    # Matplotlib fig
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower')
    mask_shape.boundary.plot(ax=ax, color='black', linewidth=0.8)
    
    cache = {'gv': gv, 'gx': gx, 'gy': gy, 'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy, 'cmap': cmap, 'norm': norm, 'custom_levels': custom_levels, 'mask_shape': mask_shape, 'shape_col': shape_col}
    return m, fig, cache, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    if 'interpol_fig' not in st.session_state: st.session_state['interpol_fig'] = None
    if 'folium_map_obj' not in st.session_state: st.session_state['folium_map_obj'] = None
    if 'folium_fig_obj' not in st.session_state: st.session_state['folium_fig_obj'] = None
    if 'interp_cache' not in st.session_state: st.session_state['interp_cache'] = None
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""
        obs_mode = ""
        
        title_interpol = ""
        data_file_interpol = None
        btn_run_interpol = False
        custom_bounds_dict = None

        if topic == "Dữ liệu quan trắc":
            if st.session_state['logged_in']:
                obs_mode = st.radio("Chọn nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy nhiệt độ", "Nội suy lượng mưa", "Nội suy linh tinh", "Đề tài Quảng Ninh"])
                
                # --- LOGIC ĐỀ TÀI QUẢNG NINH ---
                if obs_mode == "Đề tài Quảng Ninh":
                    st.markdown("---")
                    st.markdown("### 🏛️ ĐỀ TÀI QUẢNG NINH")
                    title_interpol = st.text_input("Tiêu đề bản đồ:", value="Bản đồ Nội suy Quảng Ninh")
                    data_file_interpol = st.file_uploader("Upload số liệu Quảng Ninh:", type=['xlsx', 'csv'])
                    
                    st.markdown("**Cấu hình hiển thị**")
                    cmap_list = plt.colormaps()
                    cmap_option = st.selectbox("Thang màu:", cmap_list, index=cmap_list.index('jet'))
                    
                    threshold_type = st.radio("Cách chia ngưỡng:", ["Số lớp", "Tùy chỉnh"])
                    num_bins, custom_levels = 10, None
                    if threshold_type == "Số lớp":
                        num_bins = st.number_input("Số lớp:", 2, 50, 10)
                    else:
                        cl_str = st.text_input("Ngưỡng (vd: 10,20,30):", "0,10,20,30,40,50")
                        try: custom_levels = [float(x.strip()) for x in cl_str.split(',')]
                        except: pass
                    
                    custom_bounds_dict = {'minx': 106.4, 'maxx': 108.1, 'miny': 20.5, 'maxy': 21.7}
                    btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ QUẢNG NINH", type="primary", use_container_width=True)

                # (Giữ nguyên các khối lệnh Nội suy nhiệt độ, lượng mưa, linh tinh...)
                elif obs_mode in ["Nội suy nhiệt độ", "Nội suy lượng mưa", "Nội suy linh tinh"]:
                    # ... Copy logic cũ của bạn ở đây ...
                    # (Để tiết kiệm không gian, tôi chỉ viết tóm tắt logic linh tinh bên dưới)
                    title_interpol = st.text_input("Tiêu đề bản đồ:", value="Bản đồ Nội suy")
                    data_file_interpol = st.file_uploader("Chọn file:", type=['xlsx', 'csv'], key="up_lt")
                    cmap_option = st.selectbox("Màu:", plt.colormaps(), index=0)
                    num_bins = 10; custom_levels = None
                    btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ", type="primary")

                if st.button("🔒 Đăng xuất"):
                    st.session_state['logged_in'] = False
                    st.rerun()

        # (Giữ nguyên các topic khác: Bão, Vệ tinh, KMA)
        # ...

    # --- MAIN CONTENT ---
    if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
        if obs_mode == "Đề tài Quảng Ninh":
            if btn_run_interpol:
                if data_file_interpol:
                    try:
                        # 1. Đọc dữ liệu điểm
                        df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)
                        
                        # 2. Đọc GDB lớp nền Quảng Ninh
                        if os.path.exists(QN_GDB_PATH):
                            with st.spinner("Đang đọc nền Quảng Ninh từ GDB và chuyển hệ tọa độ..."):
                                # Đọc GDB và chuyển về WGS84
                                qn_gdf = gpd.read_file(QN_GDB_PATH)
                                if qn_gdf.crs and qn_gdf.crs.to_epsg() != 4326:
                                    qn_gdf = qn_gdf.to_crs(epsg=4326)
                                
                                # Chạy nội suy với mask là Quảng Ninh
                                m_map, m_fig, m_cache, err = run_interactive_folium_interpolation(
                                    df_in, title_interpol, cmap_option, num_bins, custom_levels, 
                                    selected_provinces=None, shape_col=None, 
                                    custom_bounds=custom_bounds_dict, mask_gdf=qn_gdf
                                )
                                
                                if err: st.error(err)
                                else:
                                    st.session_state['folium_map_obj'] = m_map
                                    st.session_state['folium_fig_obj'] = m_fig
                                    st.session_state['interp_cache'] = m_cache
                        else:
                            st.error(f"Không tìm thấy file nền {QN_GDB_PATH}")
                    except Exception as e: st.error(f"Lỗi: {e}")
            
            if st.session_state['folium_map_obj']:
                st_folium(st.session_state['folium_map_obj'], width=None, height=800, use_container_width=True)
                # Nút tải xuống...
                buf = io.BytesIO()
                st.session_state['folium_fig_obj'].savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("⬇️ Tải bản đồ Quảng Ninh", buf.getvalue(), "quang_ninh.png", "image/png")

        # Các mục WeatherObs, KTTV tự động... (Giữ nguyên code cũ của bạn)
        elif "WeatherObs" in obs_mode:
            st.markdown(f'<iframe src="{LINK_WEATHEROBS}" style="width:100%; height:95vh;"></iframe>', unsafe_allow_html=True)
        elif "Gió tự động" in obs_mode:
            st.markdown(f'<iframe src="{LINK_WIND_AUTO}" style="width:100%; height:95vh;"></iframe>', unsafe_allow_html=True)

    # Khối xử lý đăng nhập (Giữ nguyên)
    elif topic == "Dữ liệu quan trắc" and not st.session_state['logged_in']:
        st.title("🔐 Đăng nhập")
        with st.form("login"):
            u = st.text_input("User"); p = st.text_input("Pass", type="password")
            if st.form_submit_button("Đăng nhập"):
                if u == "admin" and p == "kttv@2026":
                    st.session_state['logged_in'] = True; st.rerun()

    # (Các topic khác...)
    elif topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?overlay=satellite&zoom=5&lat=16&lon=114", height=1000)

if __name__ == "__main__":
    main()
