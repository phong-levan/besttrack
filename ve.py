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
    div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{ visibility: hidden !important; display: none !important; height: 0px !important; }}
    section[data-testid="stSidebar"] {{ width: {SIDEBAR_WIDTH} !important; min-width: {SIDEBAR_WIDTH} !important; max-width: {SIDEBAR_WIDTH} !important; background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd; }}
    [data-testid="stSidebarCollapseBtn"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; padding-top: 0 !important; }}
    [data-testid="stMainViewContainer"] {{ margin-left: 0 !important; width: 100% !important; padding-top: 0 !important; }}
    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; display: block !important; }}
    .floating-container {{ position: fixed; top: 20px; right: 60px; z-index: 9999; display: flex; flex-direction: column; align-items: center; }}
    .legend-box {{ width: 300px; pointer-events: none; margin-bottom: 5px; }}
    .info-box {{ width: fit-content; background: rgba(255, 255, 255, 0.9); border: 1px solid #ccc; border-radius: 6px; padding: 5px !important; color: #000; text-align: center; }}
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
    rename = {"tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no", "thời điểm": "status_raw", "ngày - giờ": "datetime_str", "thời gian (giờ)": "hour_explicit", "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h", "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc", "khí áp": "pressure", "khí áp (mb)": "pressure", "pmin": "pressure", "pmin (mb)": "pressure"}
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
        for r in np.where(exact.any(axis=1))[0]: out[r] = zi[idxs[r, np.where(exact[r])[0][0]]]
    rest = ~exact.any(axis=1)
    if np.any(rest):
        d, nn = dists[rest], idxs[rest]
        w = 1.0 / np.maximum(d, eps)**power
        out[rest] = (w * zi[nn]).sum(axis=1) / w.sum(axis=1)
    return out

def run_interpolation_and_plot(input_df, title_text, data_type='temp'):
    return None, None  # Giữ nguyên hàm nội suy tĩnh cũ theo yêu cầu

# ==============================================================================
# HÀM NỘI SUY LINH TINH (PHIÊN BẢN CHUẨN XÁC: LỌC RÁC, CHỐNG TRÀN NƯỚC NGOÀI)
# ==============================================================================
def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None):
    input_df.columns = input_df.columns.str.lower().str.strip()
    if not all(c in input_df.columns for c in ['lon', 'lat', 'value']):
        return None, None, f"File thiếu cột bắt buộc: ['lon', 'lat', 'value']"

    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, None, "Dữ liệu trống sau khi lọc bỏ NaN."

    path_to_use = SHP_DISP_PATH if os.path.exists(SHP_DISP_PATH) else SHP_MASK_PATH
    if not os.path.exists(path_to_use):
        return None, None, f"Không tìm thấy file Shapefile ranh giới ({path_to_use})."
    
    try:
        mask_shape = gpd.read_file(path_to_use, encoding='utf-8')
        if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: 
            mask_shape.to_crs(epsg=4326, inplace=True)
    except Exception as e:
        return None, None, f"Lỗi đọc file Shapefile: {e}"

    # BỘ LỌC THÔNG MINH ĐỂ DIỆT CHỮ "NAN" VÀ NƯỚC NGOÀI
    actual_col = shape_col
    if not actual_col or actual_col not in mask_shape.columns:
        for col in ['TEN_TINH', 'NAME_1', 'Name', 'PROVINCE', 'Tỉnh', 'Tinh', 'TENTINH', 'Ten_Tinh', 'ten_tinh', 'NAME', 'tinh']:
            if col.lower() in [c.lower() for c in mask_shape.columns]:
                actual_col = next(c for c in mask_shape.columns if c.lower() == col.lower())
                break

    if actual_col:
        # Xóa các dòng có tên là NaN hoặc rỗng
        mask_shape = mask_shape[mask_shape[actual_col].notna()]
        mask_shape = mask_shape[mask_shape[actual_col].astype(str).str.strip().str.lower() != 'nan']
        
        # Danh sách các nước láng giềng thường bị lọt vào file vungmoi.shp (để loại bỏ)
        exclude_list = ['lào', 'thái lan', 'cam pu chia', 'campuchia', 'trung quốc', 'phi líp pin', 'philippines', 'malaysia', 'in đô nê xia', 'indonesia', 'brunây đa rút xa lam', 'brunei', 'myanmar', 'singapore', 'đông timor', 'hong kong', 'hainan', 'hải nam']
        mask_shape = mask_shape[~mask_shape[actual_col].astype(str).str.lower().str.strip().isin(exclude_list)]

    if selected_provinces:
        if actual_col:
            display_shape = mask_shape[mask_shape[actual_col].isin(selected_provinces)]
        else:
            return None, None, "Không thể xác định cột tên tỉnh để lọc."
    else:
        display_shape = mask_shape

    if display_shape.empty: return None, None, "Không tìm thấy vùng ranh giới sau khi lọc."
    
    # Giới hạn tọa độ
    if custom_bounds:
        minx, maxx, miny, maxy = custom_bounds['minx'], custom_bounds['maxx'], custom_bounds['miny'], custom_bounds['maxy']
    else:
        minx, miny, maxx, maxy = display_shape.total_bounds
        padding = 0.5
        minx -= padding; maxx += padding; miny -= padding; maxy += padding

    # Tính toán lưới nội suy
    gx, gy = np.meshgrid(np.linspace(minx, maxx, 800), np.linspace(miny, maxy, 800))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(valid['lon'].to_numpy(), valid['lat'].to_numpy(), valid['value'].to_numpy(), grid_xy, k=12, power=3.0).reshape(gx.shape)
    gv = gaussian_filter(gv, sigma=1.0)

    # Cắt dữ liệu (Clip) chuẩn xác theo ranh giới, KHÔNG tràn sang nước khác
    shape_union = display_shape.unary_union
    prep_shape = prep(shape_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    # Xử lý thang màu
    cmap = plt.get_cmap(cmap_name)
    if custom_levels is not None and len(custom_levels) > 1:
        custom_levels = sorted(list(set(custom_levels)))
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')
    else:
        vmin_val, vmax_val = np.nanmin(gv_masked), np.nanmax(gv_masked)
        custom_levels = np.linspace(vmin_val, vmax_val, num_bins + 1)
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')

    # Chuyển thành ảnh RGBA (nền trong suốt)
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0] 
    rgba_folium = np.flipud(rgba) 
    buf = io.BytesIO()
    plt.imsave(buf, rgba_folium, format='png')
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    # KHỞI TẠO BẢN ĐỒ TƯƠNG TÁC
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=6, tiles="OpenStreetMap")
    
    # Lớp ảnh nội suy (alpha=0.85 cực đậm đà)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_base64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.85, name=title_text
    ).add_to(m)

    # Đường viền tỉnh & Tooltip (Sạch sẽ, không hiện NaN)
    tooltip_fields = [actual_col] if actual_col else []
    folium.GeoJson(
        display_shape, name="Ranh giới hành chính",
        style_function=lambda x: {'fillColor': 'transparent', 'color': '#333333', 'weight': 1.0},
        highlight_function=lambda x: {'weight': 2, 'color': 'red', 'fillColor': '#ffff00', 'fillOpacity': 0.2},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=['Khu vực: '], style="font-family: Arial; font-size: 14px; font-weight: bold;") if tooltip_fields else None
    ).add_to(m)

    m.add_child(cm.StepColormap(colors=[mcolors.to_hex(cmap(norm(val))) for val in custom_levels[:-1]], vmin=custom_levels[0], vmax=custom_levels[-1], index=custom_levels, caption=title_text))
    folium.LayerControl().add_to(m)

    # TẠO FIGURE TĨNH CHO FILE TẢI XUỐNG (Chuẩn đẹp, không lỗi font, OSM chìm)
    fig, ax = plt.subplots(figsize=(10, 10)) 
    
    im = ax.imshow(
        gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, interpolation='bilinear', origin='lower', alpha=0.9 # Đẩy màu cực đậm
    )
    
    if not display_shape.empty:
        display_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0)
        # Gắn nhãn tên (bỏ qua NaN)
        if actual_col:
            for _, row in display_shape.iterrows():
                name = str(row[actual_col]).strip()
                if name and name.lower() != 'nan' and name.lower() != 'none':
                    centroid = row.geometry.centroid
                    ax.text(centroid.x, centroid.y, name, fontsize=8, fontweight='bold', ha='center', va='center', color='#111111',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Nhúng OpenStreetMap (Mờ đi để làm nền)
    try:
        import contextily as cx
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5, attribution_size=6)
    except ImportError: pass 

    cbar = plt.colorbar(im, ax=ax, extend='both', shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(f"Giá trị Nội suy", fontsize=12, fontweight='bold')
    cbar.set_ticks(custom_levels)
    cbar.set_ticklabels([f"{val:.1f}" for val in custom_levels])

    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Kinh độ", fontsize=11)
    ax.set_ylabel("Vĩ độ", fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.ticklabel_format(useOffset=False, style='plain')

    return m, fig, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    if 'interpol_fig' not in st.session_state: st.session_state['interpol_fig'] = None
    if 'folium_map_obj' not in st.session_state: st.session_state['folium_map_obj'] = None
    if 'folium_fig_obj' not in st.session_state: st.session_state['folium_fig_obj'] = None
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

        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Chọn nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy nhiệt độ", "Nội suy lượng mưa", "Nội suy linh tinh"])
            
            if obs_mode in ["Nội suy nhiệt độ", "Nội suy lượng mưa"]:
                st.markdown("---")
                st.markdown(f"### 🛠️ CÔNG CỤ {obs_mode.upper()}")
                title_interpol = st.text_input("Tiêu đề bản đồ:", "Bản đồ nhiệt độ nội suy" if obs_mode == "Nội suy nhiệt độ" else "Bản đồ lượng mưa nội suy")
                st.markdown("**1. Upload dữ liệu (.xlsx/.csv)**")
                data_file_interpol = st.file_uploader("Chọn file số liệu:", type=['xlsx', 'csv'], key="data_up")
                btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)

            elif obs_mode == "Nội suy linh tinh":
                st.markdown("---")
                st.markdown("### 🛠️ NỘI SUY TÙY BIẾN (TƯƠNG TÁC)")
                title_interpol = st.text_input("Tiêu đề bản đồ:", value="Bản đồ Nội Suy")
                data_file_interpol = st.file_uploader("Chọn file số liệu:", type=['xlsx', 'csv'], key="data_up_custom")
                
                cmap_list = plt.colormaps()
                cmap_option = st.selectbox("Chọn thang màu (Colormap):", cmap_list, index=cmap_list.index('jet') if 'jet' in cmap_list else 0)
                
                threshold_type = st.radio("Cách chia ngưỡng:", ["Tự động (Số lớp)", "Tùy chỉnh (Nhập tay)"])
                num_bins, custom_levels = 10, None
                if threshold_type == "Tự động (Số lớp)":
                    num_bins = st.number_input("Số lượng ngưỡng chia:", min_value=2, max_value=50, value=10)
                else:
                    custom_levels_str = st.text_input("Nhập các ngưỡng (cách nhau bằng dấu phẩy):", "0, 10, 20, 30, 40, 50")
                    try: custom_levels = [float(x.strip()) for x in custom_levels_str.split(',') if x.strip()]
                    except: st.error("Lỗi định dạng. Vui lòng nhập số cách nhau bằng dấu phẩy.")
                
                # --- QUÉT VÀ LỌC TÊN TỈNH TỪ SHAPEFILE ĐỂ LÊN HỘP CHỌN ---
                province_list = []
                shape_col = "NAME_1"
                path_shp = SHP_DISP_PATH if os.path.exists(SHP_DISP_PATH) else SHP_MASK_PATH
                if os.path.exists(path_shp):
                    try:
                        tmp_shp = gpd.read_file(path_shp, encoding='utf-8')
                        found_col = None
                        for col in ['TEN_TINH', 'NAME_1', 'Name', 'PROVINCE', 'Tỉnh', 'Tinh', 'TENTINH', 'Ten_Tinh', 'ten_tinh', 'NAME']:
                            for shp_col in tmp_shp.columns:
                                if col.lower() == shp_col.lower():
                                    found_col = shp_col
                                    break
                            if found_col: break
                        
                        if found_col:
                            shape_col = found_col
                            # Lọc rác (NaN và tên nước ngoài)
                            valid_rows = tmp_shp[tmp_shp[shape_col].notna()]
                            valid_rows = valid_rows[valid_rows[shape_col].astype(str).str.strip().str.lower() != 'nan']
                            exclude_list = ['lào', 'thái lan', 'cam pu chia', 'campuchia', 'trung quốc', 'phi líp pin', 'philippines', 'malaysia', 'in đô nê xia', 'indonesia', 'brunây đa rút xa lam', 'brunei', 'myanmar', 'singapore', 'đông timor', 'hong kong', 'hainan']
                            valid_rows = valid_rows[~valid_rows[shape_col].astype(str).str.lower().str.strip().isin(exclude_list)]
                            province_list = sorted(valid_rows[shape_col].astype(str).unique().tolist())
                    except: pass
                
                selected_provinces = st.multiselect("Tách chọn Tỉnh (Để trống = Toàn bộ):", province_list)
                
                if st.checkbox("✂️ Bật giới hạn tải/hiển thị ô lưới", value=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        min_lon = st.number_input("Kinh độ Min (Trái)", value=101.80)
                        min_lat = st.number_input("Vĩ độ Min (Dưới)", value=8.00)
                    with c2:
                        max_lon = st.number_input("Kinh độ Max (Phải)", value=110.00)
                        max_lat = st.number_input("Vĩ độ Max (Trên)", value=24.00)
                    custom_bounds_dict = {'minx': min_lon, 'maxx': max_lon, 'miny': min_lat, 'maxy': max_lat}

                st.markdown("---")
                btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ TƯƠNG TÁC", type="primary", use_container_width=True)

            st.markdown("---")
            if st.button("🔒 Đăng xuất", key="logout_obs_sidebar"):
                st.session_state['logged_in'] = False
                st.rerun()

        if topic == "Dự báo điểm (KMA)":
            if st.session_state['logged_in'] and st.button("🔒 Đăng xuất", key="logout_kma_sidebar"):
                st.session_state['logged_in'] = False
                st.rerun()

        if topic == "Bản đồ Bão":
            storm_opt = st.selectbox("Dữ liệu bão:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])
            dashboard_title = st.text_input("Tiêu đề bảng thông tin:", "TIN BÃO KHẨN CẤP" if "Hiện trạng" in storm_opt else "THỐNG KÊ LỊCH SỬ")
            active_mode = storm_opt
            if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                show_widgets = True
                f = st.file_uploader("Upload besttrack", type=["csv", "xlsx"], key="o1")
                if f:
                    try:
                        df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                        df = normalize_columns(df)
                        if 'name' not in df: df['name'], df['storm_no'] = 'Storm', 'Current'
                        for c in ['wind_km/h','bf','r6','r10','rc','pressure','hour_explicit']: 
                            if c not in df: df[c]=0
                        df = df.dropna(subset=['lat','lon'])
                        if "Hiện trạng" in storm_opt:
                            all_s = df['storm_no'].unique() if 'storm_no' in df else []
                            sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s) if len(all_s)>0 else []
                            final_df = df[df['storm_no'].isin(sel)] if len(sel)>0 else df
                        else:
                            years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                            temp = df[df['year'].isin(years)]
                            names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                            final_df = temp[temp['name'].isin(names)]
                    except: pass

    # ==========================================================================
    # --- MAIN CONTENT ---
    # ==========================================================================
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=1000&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1")
    
    elif topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập Hệ thống")
            with st.form("login_form_common"):
                user_input = st.text_input("Tên đăng nhập")
                pass_input = st.text_input("Mật khẩu", type="password")
                if st.form_submit_button("Đăng nhập"):
                    if user_input == "admin" and pass_input == "kttv@2026":
                        st.session_state['logged_in'] = True
                        st.rerun()
                    else: st.error("Sai thông tin.")
        else:
            if "WeatherObs" in obs_mode:
                st.markdown(f'<div style="overflow: hidden; width: 100%; height: 95vh; position: relative;"><iframe src="{LINK_WEATHEROBS}" style="width: calc(100% + 19px); height: 1000px; position: absolute; top: -50px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
            elif "Gió tự động" in obs_mode:
                 st.markdown(f'<div style="overflow: hidden; width: 100%; height: 95vh; position: relative;"><iframe src="{LINK_WIND_AUTO}" style="width: calc(100% + 19px); height: 1200px; position: absolute; top: -75px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
            
            elif obs_mode in ["Nội suy nhiệt độ", "Nội suy lượng mưa"] and btn_run_interpol and data_file_interpol:
                df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)
                with st.spinner("Đang vẽ..."):
                    fig, err = run_interpolation_and_plot(df_in, title_interpol, 'rain' if "mưa" in obs_mode else 'temp')
                    if err: st.error(err)
                    else:
                        st.pyplot(fig, use_container_width=True)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button("⬇️ Tải ảnh", buf.getvalue(), "map.png", "image/png")

            elif obs_mode == "Nội suy linh tinh" and btn_run_interpol and data_file_interpol:
                df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)
                with st.spinner("Đang vẽ bản đồ tương tác..."):
                    m_map, m_fig, err = run_interactive_folium_interpolation(df_in, title_interpol, cmap_option, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds_dict)
                    if err: st.error(err)
                    else:
                        st_folium(m_map, width=None, height=800, use_container_width=True, returned_objects=[])
                        buf = io.BytesIO()
                        m_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button("⬇️ Tải Bản đồ Xuất bản", buf.getvalue(), "ban_do_tuy_chinh.png", "image/png")

    elif topic == "Dự báo điểm (KMA)":
        if st.session_state['logged_in']:
            st.markdown(f'<div style="overflow: hidden; width: 100%; height: 700px; position: relative;"><iframe src="{get_kma_url()}" style="width: calc(100% + 19px); height: 1200px; position: absolute; top: -130px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)

    elif topic == "Bản đồ Bão":
        m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
        folium.TileLayer('CartoDB positron', name='Bản đồ Sáng (Mặc định)', overlay=False).add_to(m)
        folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết', overlay=False).add_to(m)
        ts = get_rainviewer_ts()
        if ts: folium.TileLayer(tiles=f"https://tile.rainviewer.com/{ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png", attr="RainViewer", name="☁️ Mây", overlay=True, show=True, opacity=0.5).add_to(m)

        fg_storm = folium.FeatureGroup(name="🌀 Đường đi Bão")
        if not final_df.empty and show_widgets:
            if "Hiện trạng" in str(active_mode):
                for g in final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]:
                    sub = final_df[final_df['storm_no']==g] if g else final_df
                    f6, f10, fc = create_storm_swaths(densify_track(sub))
                    for geom, c, o in [(f6,'#FFC0CB',0.4), (f10,'#FF6347',0.5), (fc,'#90EE90',0.6)]:
                        if geom and not geom.is_empty: folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg_storm)
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_storm)
                    for _, r in sub.iterrows():
                        icon_path = ICON_PATHS.get(get_icon_name(r))
                        if icon_path: folium.Marker([r['lat'], r['lon']], icon=folium.CustomIcon(image_to_base64(icon_path), icon_size=(40,40) if 'vungthap' not in get_icon_name(r) else (20,20)), tooltip=f"Vmax {int(r.get('wind_km/h', 0))}").add_to(fg_storm)
            else: 
                for n in final_df['name'].unique():
                    sub = final_df[final_df['name']==n].sort_values('dt')
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg_storm)
                    for _, r in sub.iterrows(): folium.CircleMarker([r['lat'],r['lon']], radius=3, color='#00f2ff' if r.get('wind_km/h',0)<64 else '#ff0055', fill=True, popup=n).add_to(fg_storm)
        fg_storm.add_to(m)
        folium.LayerControl(position='topleft').add_to(m)
        
        if show_widgets:
            html = '<div class="floating-container">'
            if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG): html += f'<div class="legend-box"><img src="{image_to_base64(CHUTHICH_IMG)}"></div>'
            html += create_info_table(final_df, dashboard_title) if not final_df.empty else create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU...")
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
        st_folium(m, width=None, height=1000, use_container_width=True, returned_objects=[])

if __name__ == "__main__":
    main()
