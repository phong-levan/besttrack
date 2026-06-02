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

            if disp_shape.crs and disp_shape.crs.to_epsg() != 4326: disp_shape.to_crs(epsg=4326, inplace=True)

        except Exception: pass

    else: disp_shape = mask_shape



    if mask_shape is not None:

        prep_shape = prep(mask_shape.unary_union)

        mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)

        gv_masked = np.where(mask_flat, gv, np.nan)

    else: gv_masked = gv



    fig, ax = plt.subplots(figsize=(14, 10)) 

    ax.set_title(title_text if title_text else f'Bản đồ {unit_label}', fontsize=16)

    if disp_shape is not None: disp_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, interpolation='bilinear', origin='lower')

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, extend='both')

    cbar.set_label(unit_label, fontsize=12)

    cbar.set_ticks(levels_for_ticks)

    cbar.set_ticklabels([str(l) for l in levels_for_ticks])

    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.ticklabel_format(useOffset=False, style='plain')

    return fig, None



def generate_single_province_fig(cache, prov_name, title_text):

    mask_shape = cache.get('mask_shape')

    shape_col = cache.get('shape_col', "")

    

    if mask_shape is None or not shape_col or shape_col not in mask_shape.columns:

        return None

        

    prov_shape = mask_shape[mask_shape[shape_col] == prov_name]

    if prov_shape.empty: return None



    p_minx, p_miny, p_maxx, p_maxy = prov_shape.total_bounds

    pad_x, pad_y = (p_maxx - p_minx) * 0.1, (p_maxy - p_miny) * 0.1

    p_minx -= pad_x; p_maxx += pad_x; p_miny -= pad_y; p_maxy += pad_y

    

    grid_xy = np.column_stack([cache['gx'].ravel(), cache['gy'].ravel()])

    prep_shape = prep(prov_shape.unary_union)

    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(cache['gx'].shape)

    gv_masked = np.where(mask_flat, cache['gv'], np.nan)

    

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_title(f"{title_text}\n(Khu vực: {prov_name})", fontsize=16)

    prov_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)

    im = ax.imshow(gv_masked, extent=[cache['minx'], cache['maxx'], cache['miny'], cache['maxy']], cmap=cache['cmap'], norm=cache['norm'], interpolation='bilinear', origin='lower')

    ax.set_xlim(p_minx, p_maxx); ax.set_ylim(p_miny, p_maxy)

    

    cbar = plt.colorbar(im, ax=ax, extend='both', shrink=0.7, pad=0.02)

    cbar.set_ticks(cache['custom_levels'])

    cbar.set_ticklabels([f"{val:.1f}" for val in cache['custom_levels']])

    ax.ticklabel_format(useOffset=False, style='plain')

    ax.set_xlabel("Kinh độ"); ax.set_ylabel("Vĩ độ")

    return fig



def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None):

    shape_col = shape_col or ""

    input_df.columns = input_df.columns.str.lower().str.strip()

    if not all(c in input_df.columns for c in ['lon', 'lat', 'value']): return None, None, None, "File thiếu cột bắt buộc."

    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()

    if valid.empty: return None, None, None, "Dữ liệu trống."



    if not os.path.exists(SHP_MASK_PATH): return None, None, None, "Không tìm thấy file vn34tinh.shp"

    

    mask_shape = gpd.read_file(SHP_MASK_PATH)

    if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)

        

    if selected_provinces and shape_col and shape_col in mask_shape.columns:

        mask_shape = mask_shape[mask_shape[shape_col].isin(selected_provinces)]

        if mask_shape.empty: return None, None, None, "Không tìm thấy tỉnh đã chọn."



    disp_shape = None

    if os.path.exists(SHP_DISP_PATH):

        try:

            disp_shape = gpd.read_file(SHP_DISP_PATH)

            if disp_shape.crs and disp_shape.crs.to_epsg() != 4326: disp_shape.to_crs(epsg=4326, inplace=True)

        except Exception: pass

    

    # -------------------------------------------------------------

    # BỔ SUNG YÊU CẦU: CẮT KHUNG VÀ ZOOM KHI FIX TỌA ĐỘ

    # -------------------------------------------------------------

    if custom_bounds:

        minx, maxx, miny, maxy = custom_bounds['minx'], custom_bounds['maxx'], custom_bounds['miny'], custom_bounds['maxy']

        bbox_poly = box(minx, miny, maxx, maxy)

        

        # Cắt Shapefile theo đúng giới hạn hình chữ nhật (không hiển thị cả nước nữa)

        if not mask_shape.empty:

            mask_shape = gpd.clip(mask_shape, bbox_poly)

        if disp_shape is not None and not disp_shape.empty:

            disp_shape = gpd.clip(disp_shape, bbox_poly)

    else:

        minx, miny, maxx, maxy = mask_shape.total_bounds

        minx -= 0.5; maxx += 0.5; miny -= 0.5; maxy += 0.5



    x_pts, y_pts, z_pts = valid['lon'].to_numpy(), valid['lat'].to_numpy(), valid['value'].to_numpy()

    GRID_N, SIGMA = 800, 1.0

    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))

    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    gv = idw_knn(x_pts, y_pts, z_pts, grid_xy, k=12, power=3.0).reshape(gx.shape)

    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)



    shape_union = mask_shape.unary_union

    if shape_union.is_empty:

        # Nếu cắt bỏ hết phần đất, nội suy hiện đầy khung lưới biển

        mask_flat = np.ones(gx.shape, dtype=bool)

    else:

        prep_shape = prep(shape_union)

        mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)

        

    gv_masked = np.where(mask_flat, gv, np.nan)



    cmap = plt.get_cmap(cmap_name)

    if custom_levels is not None and len(custom_levels) > 1:

        custom_levels = sorted(list(set(custom_levels)))

        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')

    else:

        vmin_val, vmax_val = np.nanmin(gv_masked), np.nanmax(gv_masked)

        if np.isnan(vmin_val): vmin_val, vmax_val = 0, 1 # Xử lý lỗi nếu ngoài biển

        custom_levels = np.linspace(vmin_val, vmax_val, num_bins + 1)

        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')



    cache_dict = {

        'gv': gv, 'gx': gx, 'gy': gy, 'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy,

        'cmap': cmap, 'norm': norm, 'custom_levels': custom_levels, 'mask_shape': mask_shape, 'shape_col': shape_col

    }



    rgba = cmap(norm(gv_masked))

    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]

    rgba_folium = np.flipud(rgba)



    buf = io.BytesIO()

    plt.imsave(buf, rgba_folium, format='png')

    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode()



    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], tiles="CartoDB positron")

    

    # -------------------------------------------------------------

    # Ép Folium phải fit (zoom vào) đúng vị trí của Bounding Box

    # -------------------------------------------------------------

    m.fit_bounds([[miny, minx], [maxy, maxx]])

    

    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_base64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.75, name=title_text, interactive=False).add_to(m)



    if disp_shape is not None and not disp_shape.empty:

        folium.GeoJson(disp_shape, name="Ranh giới Khu vực/Quốc gia", style_function=lambda x: {'fillColor': 'transparent', 'color': '#333333', 'weight': 1.5, 'dashArray': '4, 4'}, interactive=False).add_to(m)



    tooltip_fields = [shape_col] if shape_col and shape_col in mask_shape.columns else []

    tooltip_aliases = ['Tên Tỉnh: '] if tooltip_fields else []

    

    if not mask_shape.empty:

        folium.GeoJson(

            mask_shape, name="Ranh giới Tỉnh",

            style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1.0},

            highlight_function=lambda x: {'weight': 3, 'color': 'red', 'fillColor': '#ff0000', 'fillOpacity': 0.2},

            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases) if tooltip_fields else None

        ).add_to(m)



    colormap_branca = cm.StepColormap(colors=[mcolors.to_hex(cmap(norm(val))) for val in custom_levels[:-1]], vmin=custom_levels[0], vmax=custom_levels[-1], index=custom_levels, caption=title_text)

    m.add_child(colormap_branca)

    folium.LayerControl().add_to(m)



    fig, ax = plt.subplots(figsize=(12, 10))

    ax.set_title(title_text, fontsize=16)

    if disp_shape is not None and not disp_shape.empty: disp_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0)

    if not mask_shape.empty: mask_shape.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.5, linestyle=':')

    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, interpolation='bilinear', origin='lower')

    cbar = plt.colorbar(im, ax=ax, extend='both', shrink=0.7, pad=0.02)

    cbar.set_ticks(custom_levels)

    cbar.set_ticklabels([f"{val:.1f}" for val in custom_levels])

    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    ax.ticklabel_format(useOffset=False, style='plain')

    ax.set_xlabel("Kinh độ"); ax.set_ylabel("Vĩ độ")



    return m, fig, cache_dict, None



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

                obs_mode = st.radio("Chọn nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy nhiệt độ", "Nội suy lượng mưa", "Nội suy linh tinh"])

                

                if obs_mode in ["Nội suy nhiệt độ", "Nội suy lượng mưa"]:

                    st.markdown("---")

                    st.markdown(f"### 🛠️ CÔNG CỤ {obs_mode.upper()}")

                    default_title = "Bản đồ nhiệt độ nội suy" if obs_mode == "Nội suy nhiệt độ" else "Bản đồ lượng mưa nội suy"

                    title_interpol = st.text_input("Tiêu đề bản đồ:", value=default_title)

                    st.markdown("**1. Upload dữ liệu (.xlsx/.csv)**")

                    st.caption("Cột: `stations`, `lon`, `lat`, `value`")

                    data_file_interpol = st.file_uploader("Chọn file số liệu:", type=['xlsx', 'csv'], key="data_up")

                    st.markdown("---")

                    btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)



                elif obs_mode == "Nội suy linh tinh":

                    st.markdown("---")

                    st.markdown("### 🛠️ NỘI SUY TÙY BIẾN (TƯƠNG TÁC)")

                    title_interpol = st.text_input("Tiêu đề bản đồ:", value="Bản đồ Nội Suy Tùy Chọn", key="title_custom_interp")

                    data_file_interpol = st.file_uploader("Chọn file số liệu:", type=['xlsx', 'csv', 'nc'], key="data_up_custom")

                    

                    nc_var_selected = None

                    nc_time_idx = None

                    

                    if data_file_interpol and data_file_interpol.name.endswith('.nc'):

                        try:

                            tmp_path = "sidebar_check.nc"

                            with open(tmp_path, "wb") as f: f.write(data_file_interpol.getvalue())

                            ds_tmp = None

                            for eng in ['netcdf4', 'scipy', 'h5netcdf', None]:

                                try:

                                    ds_tmp = xr.open_dataset(tmp_path, engine=eng)

                                    break

                                except: pass

                            

                            if ds_tmp is None:

                                st.error("Không thể đọc định dạng NetCDF này. Vui lòng kiểm tra lại file của bạn.")

                            else:

                                vars_list = list(ds_tmp.data_vars.keys())

                                if vars_list: nc_var_selected = st.selectbox("📌 Chọn biến dữ liệu (Variable):", vars_list)

                                time_dim = next((d for d in ds_tmp.dims if d.lower() in ['time', 't', 'valid_time']), None)

                                

                                if time_dim:

                                    time_values = ds_tmp[time_dim].values

                                    if np.issubdtype(time_values.dtype, np.datetime64):

                                        time_options = [pd.to_datetime(str(t)).strftime("%Y-%m-%d %H:%M:%S") for t in time_values]

                                    else:

                                        time_options = [str(t) for t in time_values]

                                    selected_time_str = st.selectbox("⏳ Chọn thời gian (Time Step):", time_options)

                                    nc_time_idx = time_options.index(selected_time_str)

                                else: st.info("File NetCDF không có dimension thời gian (Time).")

                                ds_tmp.close()



                            if os.path.exists(tmp_path):

                                try: os.remove(tmp_path)

                                except: pass

                            data_file_interpol.seek(0)

                        except Exception as e: st.error(f"Lỗi đọc file NetCDF: {e}")

                    

                    st.markdown("**1. Cấu hình màu & Ngưỡng**")

                    cmap_list = plt.colormaps()

                    default_cmap_idx = cmap_list.index('jet') if 'jet' in cmap_list else 0

                    cmap_option = st.selectbox("Chọn thang màu (Colormap):", cmap_list, index=default_cmap_idx)

                    

                    fig_cmap, ax_cmap = plt.subplots(figsize=(3, 0.2))

                    fig_cmap.subplots_adjust(top=1, bottom=0, left=0, right=1)

                    gradient = np.linspace(0, 1, 256).reshape(1, -1)

                    ax_cmap.imshow(gradient, aspect='auto', cmap=cmap_option)

                    ax_cmap.set_axis_off()

                    st.pyplot(fig_cmap)



                    threshold_type = st.radio("Cách chia ngưỡng:", ["Tự động (Số lớp)", "Tùy chỉnh (Nhập tay)"])

                    num_bins, custom_levels = 10, None

                    if threshold_type == "Tự động (Số lớp)":

                        num_bins = st.number_input("Số lượng ngưỡng chia:", min_value=2, max_value=50, value=10)

                    else:

                        custom_levels_str = st.text_input("Nhập các ngưỡng (cách nhau bằng dấu phẩy):", "0, 10, 20, 30, 40, 50")

                        try: custom_levels = [float(x.strip()) for x in custom_levels_str.split(',') if x.strip()]

                        except: st.error("Lỗi định dạng. Vui lòng nhập số.")

                    

                    st.markdown("**2. Ranh giới Tỉnh**")

                    province_list, shape_col = [], ""

                    if os.path.exists(SHP_MASK_PATH):

                        try:

                            tmp_shp = gpd.read_file(SHP_MASK_PATH)

                            for col in ['TEN_TINH', 'NAME_1', 'Name', 'PROVINCE', 'Tỉnh', 'Tinh', 'tentinh', 'Ten_Tinh']:

                                if col in tmp_shp.columns:

                                    shape_col = col

                                    province_list = sorted(tmp_shp[col].dropna().unique().tolist())

                                    break

                        except: pass

                    

                    selected_provinces = []

                    if province_list:

                        quick_prov = st.selectbox("Hộp chọn nhanh 1 Tỉnh (Bản đồ chính):", ["-- Tất cả Tỉnh --"] + province_list)

                        multi_provs = st.multiselect("Hoặc chọn thủ công nhiều Tỉnh:", province_list)

                        selected_provinces = [quick_prov] if quick_prov != "-- Tất cả Tỉnh --" else multi_provs

                    

                    st.markdown("**3. Cắt cúp theo Tọa độ**")

                    use_custom_bounds = st.checkbox("✂️ Giới hạn tải & hiển thị theo Tọa độ", value=False)

                    if use_custom_bounds:

                        col_b1, col_b2 = st.columns(2)

                        with col_b1: min_lon = st.number_input("Kinh độ Min", value=101.80); min_lat = st.number_input("Vĩ độ Min", value=8.00)

                        with col_b2: max_lon = st.number_input("Kinh độ Max", value=115.00); max_lat = st.number_input("Vĩ độ Max", value=24.00)

                        custom_bounds_dict = {'minx': min_lon, 'maxx': max_lon, 'miny': min_lat, 'maxy': max_lat}



                    st.markdown("---")

                    btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ TƯƠNG TÁC", type="primary", use_container_width=True)



                st.markdown("---")

                if st.button("🔒 Đăng xuất", key="logout_obs_sidebar"):

                    st.session_state['logged_in'] = False

                    st.rerun()



        if topic == "Dự báo điểm (KMA)":

            if st.session_state['logged_in']:

                st.markdown("---")

                if st.button("🔒 Đăng xuất", key="logout_kma_sidebar"):

                    st.session_state['logged_in'] = False

                    st.rerun()



        if topic == "Bản đồ Bão":

            storm_opt = st.selectbox("Dữ liệu bão:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])

            default_title = "TIN BÃO KHẨN CẤP" if "Hiện trạng" in storm_opt else "THỐNG KÊ LỊCH SỬ"

            dashboard_title = st.text_input("Tiêu đề bảng thông tin:", value=default_title)

            active_mode = storm_opt

            if "Hiện trạng" in storm_opt:

                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):

                    show_widgets = True

                    f = st.file_uploader("Upload besttrack (.csv / .xlsx)", type=["csv", "xlsx"], key="o1")

                    if f:

                        try:

                            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)

                            df = normalize_columns(df)

                            if 'name' not in df: df['name'], df['storm_no'] = 'Storm', 'Current'

                            for c in ['wind_km/h','bf','r6','r10','rc','pressure','hour_explicit']: 

                                if c not in df: df[c]=0

                            df = df.dropna(subset=['lat','lon'])

                            all_s = df['storm_no'].unique() if 'storm_no' in df else []

                            sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s) if len(all_s)>0 else []

                            final_df = df[df['storm_no'].isin(sel)] if len(sel)>0 else df

                        except: pass

                    else: st.info("Vui lòng upload file dữ liệu để xem thông tin bão.")

            else:

                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):

                    show_widgets = True

                    f = st.file_uploader("Upload besttrack_capgio.xlsx", type="xlsx", key="o2")

                    if f:

                        try:

                            df = pd.read_excel(f)

                            df = normalize_columns(df)

                            df = df.dropna(subset=['lat','lon'])

                            years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])

                            temp = df[df['year'].isin(years)]

                            names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())

                            final_df = temp[temp['name'].isin(names)]

                        except: pass

                    else: st.info("Vui lòng upload file dữ liệu lịch sử bão.")



    # --- MAIN CONTENT ---

    if topic == "Ảnh mây vệ tinh":

        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=1000&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1")

    

    elif topic == "Dữ liệu quan trắc":

        if not st.session_state['logged_in']:

            st.title("🔐 Đăng nhập Hệ thống")

            st.info("Vui lòng đăng nhập để truy cập Dữ liệu Quan trắc & Dự báo KMA.")

            with st.form("login_form_common"):

                user_input = st.text_input("Tên đăng nhập")

                pass_input = st.text_input("Mật khẩu", type="password")

                if st.form_submit_button("Đăng nhập"):

                    if user_input == "admin" and pass_input == "kttv@2026":

                        st.session_state['logged_in'] = True

                        st.success("Đăng nhập thành công!")

                        st.rerun()

                    else: st.error("Tên đăng nhập hoặc mật khẩu không đúng.")

        else:

            if "WeatherObs" in obs_mode:

                st.markdown(f'<div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;"><iframe src="{LINK_WEATHEROBS}" style="width: calc(100% + 19px); height: 1000px; position: absolute; top: -50px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)

            elif "Gió tự động" in obs_mode:

                 st.markdown(f'<div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;"><iframe src="{LINK_WIND_AUTO}" style="width: calc(100% + 19px); height: 1200px; position: absolute; top: -75px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)

            elif obs_mode in ["Nội suy nhiệt độ", "Nội suy lượng mưa"]:

                if btn_run_interpol:

                    if data_file_interpol:

                        try:

                            df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)

                            data_type = 'rain' if obs_mode == "Nội suy lượng mưa" else 'temp'

                            with st.spinner("Đang tính toán nội suy và tạo bản đồ..."):

                                fig, err = run_interpolation_and_plot(df_in, title_interpol, data_type)

                                if err: st.error(f"❌ {err}")

                                else: st.session_state['interpol_fig'] = fig

                        except Exception as e: st.error(f"❌ Lỗi: {e}")

                    else: st.toast("Vui lòng upload file dữ liệu trước!", icon="⚠️")



                if st.session_state['interpol_fig']:

                    st.pyplot(st.session_state['interpol_fig'], use_container_width=True)

                    st.markdown("### 📥 Tải xuống")

                    col_dl1, col_dl2 = st.columns([1, 3])

                    with col_dl1: fmt = st.selectbox("Định dạng:", ["png", "pdf"], key="fmt_static")

                    buf = io.BytesIO()

                    st.session_state['interpol_fig'].savefig(buf, format=fmt, dpi=300, bbox_inches='tight')

                    buf.seek(0)

                    with col_dl2:

                        st.write(""); st.write("")

                        st.download_button(label=f"⬇️ Tải ảnh về ({fmt.upper()})", data=buf, file_name=f"ban_do_tinh.{fmt}", mime=f"image/{fmt}", key="dl_btn_static")

                else: st.info("👈 Vui lòng cấu hình và nhấn nút 'VẼ BẢN ĐỒ' ở thanh menu bên trái.")



            elif obs_mode == "Nội suy linh tinh":

                if btn_run_interpol:

                    if data_file_interpol:

                        try:

                            if data_file_interpol.name.endswith('.nc'):

                                tmp_path = "run_check.nc"

                                with open(tmp_path, "wb") as f: f.write(data_file_interpol.getvalue())

                                

                                ds = None

                                for eng in ['netcdf4', 'scipy', 'h5netcdf', None]:

                                    try:

                                        ds = xr.open_dataset(tmp_path, engine=eng)

                                        break

                                    except: pass

                                

                                if ds is None:

                                    st.error("Không thể đọc file NetCDF khi vẽ. Vui lòng đảm bảo file không bị hỏng.")

                                    df_in = pd.DataFrame()

                                else:

                                    time_dim = next((d for d in ds.dims if d.lower() in ['time', 't', 'valid_time']), None)

                                    if time_dim and nc_time_idx is not None: ds = ds.isel({time_dim: nc_time_idx})

                                        

                                    var_name = nc_var_selected if nc_var_selected else list(ds.data_vars.keys())[0]

                                    df_nc = ds[var_name].to_dataframe().reset_index()

                                    

                                    lat_col = next((c for c in df_nc.columns if c.lower() in ['lat', 'latitude', 'y']), None)

                                    lon_col = next((c for c in df_nc.columns if c.lower() in ['lon', 'longitude', 'x']), None)

                                    

                                    if lat_col and lon_col:

                                        df_in = df_nc.rename(columns={lat_col: 'lat', lon_col: 'lon', var_name: 'value'})

                                        df_in = df_in[['lon', 'lat', 'value']].dropna()

                                    else:

                                        st.error("Không tìm thấy các biến tọa độ lat/lon thông dụng trong file NetCDF.")

                                        df_in = pd.DataFrame()

                                    ds.close()



                                if os.path.exists(tmp_path):

                                    try: os.remove(tmp_path)

                                    except: pass

                                data_file_interpol.seek(0)

                            else:

                                df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)

                            

                            if not df_in.empty:

                                with st.spinner("Đang xử lý nội suy tương tác và trích xuất bản vẽ..."):

                                    m_map, m_fig, m_cache, err = run_interactive_folium_interpolation(df_in, title_interpol, cmap_option, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds_dict)

                                    if err: st.error(f"❌ Lỗi: {err}")

                                    else: 

                                        st.session_state['folium_map_obj'] = m_map

                                        st.session_state['folium_fig_obj'] = m_fig

                                        st.session_state['interp_cache'] = m_cache

                        except Exception as e: st.error(f"❌ Lỗi Xử lý Dữ liệu: {e}")

                    else: st.toast("Vui lòng upload file dữ liệu trước!", icon="⚠️")



                if st.session_state['folium_map_obj']:

                    st.success("Bản đồ thành công! Kéo xuống để tải Toàn bộ khu vực, HOẶC dùng hộp chọn/click bản đồ để tải riêng từng Tỉnh.")

                    map_data = st_folium(st.session_state['folium_map_obj'], width=None, height=800, use_container_width=True, returned_objects=["last_active_drawing"])

                    

                    st.markdown("---")

                    st.markdown("### 📥 Tải bản vẽ tĩnh (Toàn bộ khu vực đã chọn)")

                    col_dl1, col_dl2 = st.columns([1, 3])

                    with col_dl1: fmt = st.selectbox("Định dạng:", ["png", "pdf"], key="fmt_folium")

                    buf = io.BytesIO()

                    st.session_state['folium_fig_obj'].savefig(buf, format=fmt, dpi=300, bbox_inches='tight')

                    buf.seek(0)

                    with col_dl2:

                        st.write(""); st.write("")

                        st.download_button(label=f"⬇️ Tải Toàn bộ ({fmt.upper()})", data=buf, file_name=f"ban_do_tong.{fmt}", mime=f"image/{fmt}", key="dl_btn_folium")



                    cache = st.session_state.get('interp_cache')

                    if cache and cache.get('mask_shape') is not None and not cache['mask_shape'].empty:

                        st.markdown("---")

                        st.markdown("### 🎯 Tải bản vẽ Tỉnh riêng lẻ (Cắt theo ranh giới tỉnh)")

                        shape_col_cache = cache.get('shape_col', "")

                        

                        if not shape_col_cache or shape_col_cache not in cache['mask_shape'].columns:

                            st.warning("⚠️ Không thể tải riêng từng tỉnh vì Shapefile (vn34tinh.shp) không có cột chứa tên Tỉnh. Tính năng này tạm thời bị vô hiệu hóa.")

                        else:

                            available_provs = sorted(cache['mask_shape'][shape_col_cache].dropna().unique().tolist())

                            col_sel1, col_sel2 = st.columns([2, 2])

                            with col_sel1: selected_dl_prov = st.selectbox("Hộp chọn Tỉnh muốn tải:", ["-- Click trên bản đồ hoặc Chọn tại đây --"] + available_provs)

                            

                            clicked_prov = None

                            if map_data and map_data.get("last_active_drawing"):

                                props = map_data["last_active_drawing"].get("properties", {})

                                if shape_col_cache in props:

                                    clicked_prov = props[shape_col_cache]

                                    with col_sel2:

                                        st.write(""); st.info(f"💡 Đang click chọn: **{clicked_prov}** trên bản đồ.")

                            

                            final_dl_prov = selected_dl_prov if selected_dl_prov != "-- Click trên bản đồ hoặc Chọn tại đây --" else clicked_prov

                            if final_dl_prov:

                                prov_fig = generate_single_province_fig(cache, final_dl_prov, st.session_state.get("title_custom_interp", "Bản đồ Nội Suy"))

                                if prov_fig:

                                    col_p1, col_p2 = st.columns([1, 3])

                                    with col_p1: fmt_prov = st.selectbox("Định dạng ảnh:", ["png", "pdf"], key="fmt_prov")

                                    buf_prov = io.BytesIO()

                                    prov_fig.savefig(buf_prov, format=fmt_prov, dpi=300, bbox_inches='tight')

                                    buf_prov.seek(0)

                                    with col_p2:

                                        st.write(""); st.write("")

                                        st.download_button(label=f"⬇️ Tải ảnh Tỉnh {final_dl_prov} ({fmt_prov.upper()})", data=buf_prov, file_name=f"ban_do_{final_dl_prov}.{fmt_prov}", mime=f"image/{fmt_prov}", key="dl_btn_prov")

                else: st.info("👈 Vui lòng cấu hình dữ liệu, chọn màu, ngưỡng, tọa độ và nhấn 'VẼ BẢN ĐỒ TƯƠNG TÁC'.")



    elif topic == "Dự báo điểm (KMA)":

        if not st.session_state['logged_in']:

            st.title("🔐 Đăng nhập Hệ thống")

            st.info("Vui lòng đăng nhập để truy cập Dữ liệu Quan trắc & Dự báo KMA.")

            with st.form("login_form_common_kma"):

                user_input = st.text_input("Tên đăng nhập")

                pass_input = st.text_input("Mật khẩu", type="password")

                if st.form_submit_button("Đăng nhập"):

                    if user_input == "admin" and pass_input == "kttv@2026":

                        st.session_state['logged_in'] = True

                        st.success("Đăng nhập thành công!")

                        st.rerun()

                    else: st.error("Tên đăng nhập hoặc mật khẩu không đúng.")

        else:

            realtime_kma_url = get_kma_url()

            st.markdown(f'<div style="overflow: hidden; width: 100%; height: 700px; position: relative; border: 1px solid #ddd;"><iframe src="{realtime_kma_url}" style="width: calc(100% + 19px); height: 1200px; position: absolute; top: -130px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)

            st.caption(f"Đang hiển thị dữ liệu từ nguồn KMA (Hàn Quốc). Link gốc: {realtime_kma_url}")



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

                    

                    for _, r in sub.iterrows():

                        icon_key = get_icon_name(r)

                        icon_path = ICON_PATHS.get(icon_key)

                        icon_base64 = image_to_base64(icon_path) if icon_path else None

                        if icon_base64:

                            i_size, i_anchor = ((20, 20), (10, 10)) if 'vungthap' in icon_key else ((40, 40), (20, 20))

                            icon = folium.CustomIcon(icon_image=icon_base64, icon_size=i_size, icon_anchor=i_anchor)

                            folium.Marker(location=[r['lat'], r['lon']], icon=icon, tooltip=f"Vmax {int(r.get('wind_km/h', 0))} km/h").add_to(fg_storm)

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

            html_to_render = '<div class="floating-container">'

            if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG):

                with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()

                html_to_render += f'<div class="legend-box"><img src="data:image/png;base64,{b64}"></div>'

            html_to_render += create_info_table(final_df, dashboard_title) if not final_df.empty else create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU...")

            html_to_render += '</div>'

            st.markdown(html_to_render, unsafe_allow_html=True)

        

        st_folium(m, width=None, height=1000, use_container_width=True)



if __name__ == "__main__":

    main() 

