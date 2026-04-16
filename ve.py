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
# 0. CẤU HÌNH FONT TIẾNG VIỆT (TRÁNH LỖI Ô VUÔNG KHI XUẤT ẢNH)
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

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

# Màu sắc
COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
COLOR_TEXT = "#333333"
COLOR_ACCENT = "#007bff"
COLOR_BORDER = "#dee2e6"
SIDEBAR_WIDTH = "300px"

# Cấu hình trang
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
    .block-container {{
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }}
    header, footer {{ display: none !important; }}
    
    div[data-testid="stToolbar"], 
    div[data-testid="stDecoration"], 
    div[data-testid="stStatusWidget"] {{
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }}

    section[data-testid="stSidebar"] {{
        display: block !important;
        visibility: visible !important;
        width: {SIDEBAR_WIDTH} !important;
        min-width: {SIDEBAR_WIDTH} !important;
        max-width: {SIDEBAR_WIDTH} !important;
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        height: 100vh !important;
        transform: none !important;
        z-index: 100000 !important;
        background-color: {COLOR_SIDEBAR} !important;
        border-right: 1px solid #ddd;
    }}

    [data-testid="stSidebarCollapseBtn"],
    [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}

    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; padding-top: 0 !important; }}
    [data-testid="stMainViewContainer"] {{ margin-left: 0 !important; width: 100% !important; padding-top: 0 !important; }}

    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; display: block !important; }}

    .floating-container {{
        position: fixed; top: 20px; right: 60px; z-index: 9999;
        display: flex; flex-direction: column; align-items: center;    
    }}

    .legend-box {{ width: 300px; pointer-events: none; margin-bottom: 5px; }}
    
    .info-box {{
        width: fit-content; background: rgba(255, 255, 255, 0.9);
        border: 1px solid #ccc; border-radius: 6px;
        padding: 5px !important;
        color: #000; text-align: center;
    }}
    
    .info-box table {{ width: 100%; margin: 0 auto; border-collapse: collapse; }}
    
    .info-box th, .info-box td {{ 
        text-align: center !important; 
        padding: 2px 5px !important; 
        font-size: 12px !important; 
    }}
    
    .info-title {{ 
        font-weight: bold; 
        margin-bottom: 2px; 
        font-size: 14px !important; 
    }}
    
    .info-subtitle {{ 
        font-size: 10px !important; 
        margin-bottom: 5px; 
        font-style: italic; 
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
    if not os.path.exists(image_path): return None
    with open(image_path, "rb") as f: encoded = base64.b64encode(f.read()).decode()
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

# === LOGIC NỘI SUY ===
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
    minx, maxx = 101.8, 115.0
    miny, maxy = 8.0, 23.9
    GRID_N = 1000 
    SIGMA = 1.5
    IDW_POWER = 3.0
    KNN = 12

    if data_type == 'rain':
        vmin, vmax = 0, 1400
        levels_for_ticks = np.arange(0, 1450, 100)
        colors = ['#FFFFFF', '#A0E6FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800080', '#4B0082']
        cmap = LinearSegmentedColormap.from_list('rain_smooth', colors, N=512)
        cmap.set_under(colors[0])
        cmap.set_over(colors[-1])
        unit_label = "Lượng mưa (mm)"
    else: 
        vmin, vmax = 0.0, 40.0
        levels_for_ticks = list(range(0, 42, 4))
        colors = [(0.0, '#FFFFFF'), (0.1, '#D0F0FF'), (0.2, '#00A0FF'), (0.4, '#00FF00'),
                 (0.6, '#FFFF00'), (0.75, '#FFA500'), (0.9, '#FF0000'), (1.0, '#8B0000')]
        cmap = LinearSegmentedColormap.from_list("custom_smooth_temp", colors, N=256)
        unit_label = "Nhiệt độ (°C)"

    norm = Normalize(vmin=vmin, vmax=vmax)

    input_df.columns = input_df.columns.str.lower().str.strip()
    cols_check = ['lon', 'lat', 'value']
    if not all(c in input_df.columns for c in cols_check):
        return None, f"File thiếu cột bắt buộc: {cols_check}"

    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty:
        return None, "Dữ liệu trống sau khi lọc bỏ NaN."

    x_pts = valid['lon'].to_numpy()
    y_pts = valid['lat'].to_numpy()
    z_pts = valid['value'].to_numpy()

    edge_points = pd.DataFrame({
        'lon': [minx, minx, maxx, maxx, (minx + maxx)/2],
        'lat': [miny, maxy, miny, maxy, (miny + maxy)/2],
        'value': [float(np.nanmean(z_pts))] * 5
    })
    
    aug = pd.concat([valid[['lon', 'lat', 'value']], edge_points], ignore_index=True)
    xi = aug['lon'].to_numpy()
    yi = aug['lat'].to_numpy()
    zi = aug['value'].to_numpy()

    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    gv = idw_knn(xi, yi, zi, grid_xy, k=KNN, power=IDW_POWER).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    mask_shape = None
    disp_shape = None
    
    if os.path.exists(SHP_MASK_PATH):
        try:
            mask_shape = gpd.read_file(SHP_MASK_PATH, encoding='utf-8')
            if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)
        except Exception as e: return None, f"Lỗi đọc Mask Shapefile: {e}"
    else:
        bbox_poly = box(minx, miny, maxx, maxy)
        mask_shape = gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs='EPSG:4326')

    if os.path.exists(SHP_DISP_PATH):
        try:
            disp_shape = gpd.read_file(SHP_DISP_PATH, encoding='utf-8')
            if disp_shape.crs and disp_shape.crs.to_epsg() != 4326: disp_shape.to_crs(epsg=4326, inplace=True)
        except Exception as e: return None, f"Lỗi đọc Display Shapefile: {e}"
    else:
        disp_shape = mask_shape

    if mask_shape is not None:
        shape_union = mask_shape.unary_union
        prep_shape = prep(shape_union)
        mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
        gv_masked = np.where(mask_flat, gv, np.nan)
    else:
        gv_masked = gv

    fig, ax = plt.subplots(figsize=(14, 10)) 
    ax.set_title(title_text if title_text else f'Bản đồ {unit_label}', fontsize=16)

    if disp_shape is not None:
        disp_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    im = ax.imshow(
        gv_masked,
        extent=[minx, maxx, miny, maxy],
        cmap=cmap,
        norm=norm,
        interpolation='bilinear',
        origin='lower'
    )

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, extend='both')
    cbar.set_label(unit_label, fontsize=12)
    cbar.set_ticks(levels_for_ticks)
    cbar.set_ticklabels([str(l) for l in levels_for_ticks])

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.ticklabel_format(useOffset=False, style='plain')
    
    return fig, None

def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None):
    # 1. Chuẩn hóa dữ liệu đầu vào
    input_df.columns = input_df.columns.str.lower().str.strip()
    cols_check = ['lon', 'lat', 'value']
    if not all(c in input_df.columns for c in cols_check):
        return None, None, f"File thiếu cột bắt buộc: {cols_check}"

    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, None, "Dữ liệu trống sau khi lọc bỏ NaN."

    # 2. Đọc và lọc ranh giới hành chính
    if not os.path.exists(SHP_MASK_PATH):
        return None, None, "Không tìm thấy file Shapefile ranh giới."
    
    try:
        mask_shape = gpd.read_file(SHP_MASK_PATH, encoding='utf-8')
        if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: 
            mask_shape.to_crs(epsg=4326, inplace=True)
    except Exception as e:
        return None, None, f"Lỗi đọc file Shapefile: {e}"

    # Nhận diện cột THÔNG MINH (Auto-detect column)
    actual_col = None
    if shape_col in mask_shape.columns:
        actual_col = shape_col
    else:
        # Quét ưu tiên các tên cột hay gặp
        common_cols = ['TEN_TINH', 'NAME_1', 'Name', 'PROVINCE', 'Tỉnh', 'Tinh', 'TENTINH', 'Ten_Tinh', 'ten_tinh', 'NAME', 'tinh']
        for col in common_cols:
            for shp_col in mask_shape.columns:
                if col.lower() == shp_col.lower():
                    actual_col = shp_col
                    break
            if actual_col: break
            
        # Nếu vẫn không thấy, tự quét dữ liệu bên trong để đối chiếu với danh sách tỉnh đã chọn
        if not actual_col and selected_provinces:
            for col in mask_shape.columns:
                if mask_shape[col].dtype == 'object' or str(mask_shape[col].dtype) == 'string':
                    vals = mask_shape[col].dropna().astype(str).values
                    # Nếu thấy bất kỳ tỉnh nào trùng khớp, chốt luôn cột đó
                    if any(p in vals for p in selected_provinces):
                        actual_col = col
                        break

    if selected_provinces:
        if actual_col:
            display_shape = mask_shape[mask_shape[actual_col].isin(selected_provinces)]
            shape_col = actual_col # Cập nhật để tooltip hiển thị đúng
        else:
            return None, None, f"Lỗi Shapefile: Không thể tự động xác định được cột chứa tên tỉnh. Các cột hiện có: {list(mask_shape.columns)}"
    else:
        display_shape = mask_shape
        if actual_col: shape_col = actual_col

    if display_shape.empty: return None, None, "Không tìm thấy vùng ranh giới sau khi lọc."
    
    # Xác định giới hạn tọa độ
    if custom_bounds:
        minx = custom_bounds['minx']
        maxx = custom_bounds['maxx']
        miny = custom_bounds['miny']
        maxy = custom_bounds['maxy']
    else:
        minx, miny, maxx, maxy = display_shape.total_bounds
        # Mở rộng nhẹ ranh giới để lưới nội suy phủ kín biên giới
        padding = 0.5
        minx -= padding; maxx += padding; miny -= padding; maxy += padding

    x_pts = valid['lon'].to_numpy()
    y_pts = valid['lat'].to_numpy()
    z_pts = valid['value'].to_numpy()

    # 3. Tính toán nội suy IDW
    GRID_N = 800
    SIGMA = 1.0

    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    gv = idw_knn(x_pts, y_pts, z_pts, grid_xy, k=12, power=3.0).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    # 4. Cắt dữ liệu (Clip) theo đúng ranh giới (KHÔNG CHE MỜ THẾ GIỚI TRÊN FOLIUM)
    shape_union = display_shape.unary_union
    prep_shape = prep(shape_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    # 5. Xử lý thang màu
    cmap = plt.get_cmap(cmap_name)
    if custom_levels is not None and len(custom_levels) > 1:
        custom_levels = sorted(list(set(custom_levels)))
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')
    else:
        vmin_val, vmax_val = np.nanmin(gv_masked), np.nanmax(gv_masked)
        custom_levels = np.linspace(vmin_val, vmax_val, num_bins + 1)
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')

    # Chuyển đổi thành ảnh RGBA cho Folium
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0] # Trong suốt ngoài ranh giới
    rgba_folium = np.flipud(rgba) 

    buf = io.BytesIO()
    plt.imsave(buf, rgba_folium, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()

    # 6. Khởi tạo bản đồ tương tác (Nền OpenStreetMap)
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

    # Lớp ảnh nội suy
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_base64}",
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=0.85,
        name=title_text,
        interactive=False
    ).add_to(m)

    # Đường viền tỉnh - Hỗ trợ Click/Hover (Chỉnh lại Tooltip dễ nhìn)
    tooltip_fields = [shape_col] if shape_col and shape_col in display_shape.columns else []
    tooltip_aliases = ['Tên Tỉnh/Thành: '] if tooltip_fields else []
    
    folium.GeoJson(
        display_shape,
        name="Ranh giới hành chính",
        style_function=lambda x: {'fillColor': 'transparent', 'color': '#333333', 'weight': 1.0},
        highlight_function=lambda x: {'weight': 2, 'color': 'red', 'fillColor': '#ffff00', 'fillOpacity': 0.2},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields, 
            aliases=tooltip_aliases,
            style="font-family: Arial; font-size: 14px; font-weight: bold; background-color: white;"
        ) if tooltip_fields else None
    ).add_to(m)

    # Chú giải màu
    colormap_branca = cm.StepColormap(
        colors=[mcolors.to_hex(cmap(norm(val))) for val in custom_levels[:-1]],
        vmin=custom_levels[0], vmax=custom_levels[-1],
        index=custom_levels,
        caption=title_text
    )
    m.add_child(colormap_branca)
    folium.LayerControl().add_to(m)

    # 7. Tạo Figure tĩnh cho file tải (PNG/PDF)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    
    if not display_shape.empty:
        display_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        
        # Thêm nhãn tên tỉnh (Fix lỗi font ô vuông)
        if shape_col and shape_col in display_shape.columns:
            for _, row in display_shape.iterrows():
                centroid = row.geometry.centroid
                name = str(row[shape_col])
                ax.text(centroid.x, centroid.y, name, fontsize=8, fontweight='bold',
                        ha='center', va='center', color='#2b2b2b',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # Vẽ nội suy
    im = ax.imshow(
        gv_masked,
        extent=[minx, maxx, miny, maxy],
        cmap=cmap,
        norm=norm,
        interpolation='bilinear',
        origin='lower',
        alpha=0.85
    )
    
    # Nền Contextily OpenStreetMap (sẽ hoạt động nếu đã cài pip install contextily)
    try:
        import contextily as cx
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.6)
    except ImportError:
        pass # Nếu chưa cài contextily thì bỏ qua, xuất nền trắng mặc định
    
    cbar = plt.colorbar(im, ax=ax, extend='both', shrink=0.6, pad=0.02)
    cbar.set_label(f"Giá trị Nội suy", fontsize=11, fontweight='bold')
    cbar.set_ticks(custom_levels)
    cbar.set_ticklabels([f"{val:.1f}" for val in custom_levels])

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel("Kinh độ", fontsize=10)
    ax.set_ylabel("Vĩ độ", fontsize=10)

    return m, fig, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    if 'interpol_fig' not in st.session_state:
        st.session_state['interpol_fig'] = None
    if 'folium_map_obj' not in st.session_state:
        st.session_state['folium_map_obj'] = None
    if 'folium_fig_obj' not in st.session_state:
        st.session_state['folium_fig_obj'] = None
        
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

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
                obs_mode = st.radio("Chọn nguồn dữ liệu:", 
                                  ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy nhiệt độ", "Nội suy lượng mưa", "Nội suy linh tinh"])
                
                # --- MENU NỘI SUY TĨNH ---
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

                # --- MENU NỘI SUY TƯƠNG TÁC (FOLIUM) ---
                elif obs_mode == "Nội suy linh tinh":
                    st.markdown("---")
                    st.markdown("### 🛠️ NỘI SUY TÙY BIẾN (TƯƠNG TÁC)")
                    title_interpol = st.text_input("Tiêu đề bản đồ:", value="Bản đồ Nội Suy")
                    data_file_interpol = st.file_uploader("Chọn file số liệu:", type=['xlsx', 'csv'], key="data_up_custom")
                    
                    st.markdown("**1. Cấu hình màu & Ngưỡng**")
                    cmap_list = plt.colormaps()
                    default_cmap_idx = cmap_list.index('jet') if 'jet' in cmap_list else 0
                    cmap_option = st.selectbox("Chọn thang màu (Colormap):", cmap_list, index=default_cmap_idx)
                    
                    threshold_type = st.radio("Cách chia ngưỡng:", ["Tự động (Số lớp)", "Tùy chỉnh (Nhập tay)"])
                    num_bins = 10
                    custom_levels = None
                    if threshold_type == "Tự động (Số lớp)":
                        num_bins = st.number_input("Số lượng ngưỡng chia:", min_value=2, max_value=50, value=10)
                    else:
                        custom_levels_str = st.text_input("Nhập các ngưỡng (cách nhau bằng dấu phẩy):", "0, 10, 20, 30, 40, 50")
                        try:
                            custom_levels = [float(x.strip()) for x in custom_levels_str.split(',') if x.strip()]
                        except:
                            st.error("Lỗi định dạng. Vui lòng nhập số cách nhau bằng dấu phẩy.")
                    
                    st.markdown("**2. Ranh giới Tỉnh**")
                    
                    # Danh sách dự phòng để hộp chọn LUÔN LUÔN hiện lên đầy đủ
                    fallback_provinces = [
                        "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh", "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau", "Cần Thơ", "Cao Bằng", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương", "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An", "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang", "TP Hồ Chí Minh", "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
                    ]
                    province_list = fallback_provinces
                    shape_col = "NAME_1"
                    
                    # Cố gắng quét file Shapefile để lấy đúng danh sách và tên cột
                    if os.path.exists(SHP_MASK_PATH):
                        try:
                            tmp_shp = gpd.read_file(SHP_MASK_PATH, encoding='utf-8')
                            found = False
                            for col in ['TEN_TINH', 'NAME_1', 'Name', 'PROVINCE', 'Tỉnh', 'Tinh', 'TENTINH', 'Ten_Tinh', 'ten_tinh', 'NAME', 'tinh']:
                                for shp_col in tmp_shp.columns:
                                    if col.lower() == shp_col.lower():
                                        shape_col = shp_col
                                        extracted = sorted(tmp_shp[shp_col].dropna().astype(str).unique().tolist())
                                        if extracted:
                                            province_list = extracted
                                        found = True
                                        break
                                if found: break
                                
                            if not found:
                                # Fallback: tìm cột dạng string có nhiều giá trị unique
                                for col in tmp_shp.columns:
                                    if tmp_shp[col].dtype == 'object' or str(tmp_shp[col].dtype) == 'string':
                                        unique_vals = tmp_shp[col].dropna().astype(str).unique()
                                        if len(unique_vals) > 5: 
                                            shape_col = col
                                            province_list = sorted(unique_vals.tolist())
                                            break
                        except: pass
                    
                    # Hộp chọn Multiselect
                    selected_provinces = st.multiselect("Tách chọn Tỉnh (Để trống = Toàn bộ):", province_list)
                    
                    st.markdown("**3. Cắt cúp hiển thị & tải (Kinh Vĩ độ)**")
                    use_custom_bounds = st.checkbox("✂️ Bật giới hạn tải/hiển thị ô lưới", value=False)
                    if use_custom_bounds:
                        col_b1, col_b2 = st.columns(2)
                        with col_b1:
                            min_lon = st.number_input("Kinh độ Min (Trái)", value=101.80, format="%.2f")
                            min_lat = st.number_input("Vĩ độ Min (Dưới)", value=8.00, format="%.2f")
                        with col_b2:
                            max_lon = st.number_input("Kinh độ Max (Phải)", value=110.00, format="%.2f")
                            max_lat = st.number_input("Vĩ độ Max (Trên)", value=24.00, format="%.2f")
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
                    else:
                        st.info("Vui lòng upload file dữ liệu để xem thông tin bão.")
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
                    else:
                        st.info("Vui lòng upload file dữ liệu lịch sử bão.")

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
                submitted = st.form_submit_button("Đăng nhập")
                
                if submitted:
                    if user_input == "admin" and pass_input == "kttv@2026":
                        st.session_state['logged_in'] = True
                        st.success("Đăng nhập thành công!")
                        st.rerun()
                    else:
                        st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
        else:
            if "WeatherObs" in obs_mode:
                html_weather = f"""
                <div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;">
                    <iframe 
                        src="{LINK_WEATHEROBS}" 
                        style="
                            width: calc(100% + 19px); 
                            height: 1000px; 
                            position: absolute; 
                            top: -50px; 
                            left: 0px; 
                            border: none;"
                        allow="fullscreen"
                    ></iframe>
                </div>
                """
                st.markdown(html_weather, unsafe_allow_html=True)

            elif "Gió tự động" in obs_mode:
                 html_kttv = f"""
                <div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;">
                    <iframe 
                        src="{LINK_WIND_AUTO}" 
                        style="
                            width: calc(100% + 19px); 
                            height: 1200px; 
                            position: absolute; 
                            top: -75px; 
                            left: 0px; 
                            border: none;"
                        allow="fullscreen"
                    ></iframe>
                </div>
                 """
                 st.markdown(html_kttv, unsafe_allow_html=True)
            
            # --- VIEW NỘI SUY TĨNH ---
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
                else:
                    st.info("👈 Vui lòng cấu hình và nhấn nút 'VẼ BẢN ĐỒ' ở thanh menu bên trái.")

            # --- VIEW NỘI SUY TƯƠNG TÁC ---
            elif obs_mode == "Nội suy linh tinh":
                if btn_run_interpol:
                    if data_file_interpol:
                        try:
                            df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)
                            
                            with st.spinner("Đang xử lý nội suy tương tác và trích xuất bản vẽ..."):
                                m_map, m_fig, err = run_interactive_folium_interpolation(
                                    df_in, title_interpol, cmap_option, 
                                    num_bins, custom_levels, selected_provinces, shape_col, custom_bounds_dict
                                )
                                if err: 
                                    st.error(f"❌ Lỗi: {err}")
                                else: 
                                    st.session_state['folium_map_obj'] = m_map
                                    st.session_state['folium_fig_obj'] = m_fig
                        except Exception as e: 
                            st.error(f"❌ Lỗi Xử lý Dữ liệu: {e}")
                    else: 
                        st.toast("Vui lòng upload file dữ liệu trước!", icon="⚠️")

                if st.session_state['folium_map_obj']:
                    st.success("Tạo bản đồ thành công! Kéo xuống dưới cùng để TẢI ẢNH.")
                    # Render bản đồ Folium
                    st_folium(st.session_state['folium_map_obj'], width=None, height=800, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### 📥 Tải bản vẽ tĩnh (Cắt theo khu vực đã chọn)")
                    col_dl1, col_dl2 = st.columns([1, 3])
                    with col_dl1: 
                        fmt = st.selectbox("Định dạng:", ["png", "pdf"], key="fmt_folium")
                    buf = io.BytesIO()
                    st.session_state['folium_fig_obj'].savefig(buf, format=fmt, dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    with col_dl2:
                        st.write(""); st.write("")
                        st.download_button(label=f"⬇️ Tải ảnh về ({fmt.upper()})", data=buf, file_name=f"ban_do_tuy_chinh.{fmt}", mime=f"image/{fmt}", key="dl_btn_folium")
                else:
                    st.info("👈 Vui lòng cấu hình dữ liệu, chọn màu, ngưỡng, tọa độ và nhấn 'VẼ BẢN ĐỒ TƯƠNG TÁC'.")

    elif topic == "Dự báo điểm (KMA)":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập Hệ thống")
            st.info("Vui lòng đăng nhập để truy cập Dữ liệu Quan trắc & Dự báo KMA.")
            
            with st.form("login_form_common_kma"):
                user_input = st.text_input("Tên đăng nhập")
                pass_input = st.text_input("Mật khẩu", type="password")
                submitted = st.form_submit_button("Đăng nhập")
                
                if submitted:
                    if user_input == "admin" and pass_input == "kttv@2026":
                        st.session_state['logged_in'] = True
                        st.success("Đăng nhập thành công!")
                        st.rerun()
                    else:
                        st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
        else:
            realtime_kma_url = get_kma_url()
            html_kma = f"""
            <div style="overflow: hidden; width: 100%; height: 700px; position: relative; border: 1px solid #ddd;">
                <iframe
                    src="{realtime_kma_url}" 
                    style="
                        width: calc(100% + 19px); 
                        height: 1200px; 
                        position: absolute; 
                        top: -130px; 
                        left: 0px; 
                        border: none;"
                    allow="fullscreen"
                ></iframe>
            </div>
            """
            st.markdown(html_kma, unsafe_allow_html=True)
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
                        if geom and not geom.is_empty:
                           folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg_storm)
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_storm)
                    
                    for _, r in sub.iterrows():
                        icon_key = get_icon_name(r)
                        icon_path = ICON_PATHS.get(icon_key)
                        icon_base64 = None
                        if icon_path:
                            icon_base64 = image_to_base64(icon_path)
                        
                        if icon_base64:
                            if 'vungthap' in icon_key:
                                i_size = (20, 20)
                                i_anchor = (10, 10)
                            else:
                                i_size = (40, 40)
                                i_anchor = (20, 20)
                            
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
            
            if not final_df.empty: 
                html_to_render += create_info_table(final_df, dashboard_title)
            else: 
                html_to_render += create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU...")
            
            html_to_render += '</div>'
            st.markdown(html_to_render, unsafe_allow_html=True)
        
        st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
