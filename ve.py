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
import matplotlib.lines as mlines
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
# 0. CẤU HÌNH FONT CHO VIỆT NAM (SỬA LỖI FONT TRÊN BIỂU ĐỒ TĨNH)
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Tahoma']
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

COLOR_SIDEBAR = "#f8f9fa"
SIDEBAR_WIDTH = "300px"

st.set_page_config(page_title="Hệ thống giám sát", layout="wide", initial_sidebar_state="expanded")

# --- CSS CHUNG ---
st.markdown(f"""
    <style>
    .block-container {{ padding: 0 !important; margin: 0 !important; max-width: 100% !important; }}
    header, footer {{ display: none !important; }}
    div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{
        visibility: hidden !important; display: none !important; height: 0px !important;
    }}
    section[data-testid="stSidebar"] {{
        width: {SIDEBAR_WIDTH} !important; min-width: {SIDEBAR_WIDTH} !important;
        background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd;
    }}
    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; }}
    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; }}
    .floating-container {{ position: fixed; top: 20px; right: 60px; z-index: 9999; display: flex; flex-direction: column; align-items: center; }}
    .info-box {{ background: rgba(255, 255, 255, 0.9); border: 1px solid #ccc; border-radius: 6px; padding: 5px; color: #000; text-align: center; }}
    </style>
""", unsafe_allow_html=True)

# --- CÁC HÀM XỬ LÝ DỮ LIỆU CŨ (GIỮ NGUYÊN) ---
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
    return df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

def idw_knn(xi, yi, zi, query_xy, k=12, power=3.0):
    tree = cKDTree(np.column_stack([xi, yi]))
    dists, idxs = tree.query(query_xy, k=min(k, xi.size))
    if dists.ndim == 1: dists, idxs = dists[:, None], idxs[:, None]
    w = 1.0 / np.maximum(dists, 1e-12)**power
    return (w * zi[idxs]).sum(axis=1) / w.sum(axis=1)

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

def run_interpolation_and_plot(input_df, title_text, data_type='temp'):
    # Hàm tĩnh cũ (Giữ nguyên)
    return None, None

# ==============================================================================
# HÀM NỘI SUY LINH TINH (TÍCH HỢP OSM NỀN VÀ CẮT CHUẨN LÃNH THỔ VN)
# ==============================================================================
def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None):
    input_df.columns = input_df.columns.str.lower().str.strip()
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, None, "Dữ liệu trống."

    # 1. Nạp ranh giới và tự động tải GeoJSON VN chuẩn nếu không có file local
    path_to_use = SHP_DISP_PATH if os.path.exists(SHP_DISP_PATH) else SHP_MASK_PATH
    try:
        if os.path.exists(path_to_use):
            mask_shape = gpd.read_file(path_to_use)
        else:
            st.toast("Đang tự động tải dữ liệu ranh giới Việt Nam chuẩn...")
            url_vn = "https://raw.githubusercontent.com/TungTh/tungth.github.io/master/data/vn-provinces.json"
            mask_shape = gpd.read_file(url_vn)
            shape_col = "Name" # Tên cột của file GeoJSON tải về
            
        if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: 
            mask_shape.to_crs(epsg=4326, inplace=True)
    except Exception as e:
        return None, None, f"Lỗi ranh giới: {e}"

    # 2. Lọc ranh giới theo tỉnh đã chọn
    actual_col = shape_col
    if selected_provinces:
        # Tìm cột đúng chứa tên tỉnh
        actual_col = next((c for c in mask_shape.columns if any(p in mask_shape[c].astype(str).values for p in selected_provinces)), shape_col)
        mask_shape = mask_shape[mask_shape[actual_col].isin(selected_provinces)]

    # Hợp nhất ranh giới để tạo thành khung cắt (Khuôn đúc)
    vietnam_union = mask_shape.unary_union
    minx, miny, maxx, maxy = (custom_bounds['minx'], custom_bounds['miny'], custom_bounds['maxx'], custom_bounds['maxy']) if custom_bounds else vietnam_union.bounds

    # Tính toán nội suy
    GRID_N = 800
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(valid['lon'].values, valid['lat'].values, valid['value'].values, grid_xy).reshape(gx.shape)
    gv = gaussian_filter(gv, sigma=1.0)

    # 3. KỸ THUẬT CLIPPING: Đảm bảo màu chỉ nằm gọn 100% trong ranh giới
    prep_shape = prep(vietnam_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    # Thang màu
    cmap = plt.get_cmap(cmap_name)
    if custom_levels:
        norm = BoundaryNorm(sorted(list(set(custom_levels))), ncolors=cmap.N, extend='both')
    else:
        norm = BoundaryNorm(np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), num_bins + 1), ncolors=cmap.N, extend='both')

    # Tạo ảnh RGBA cho Folium
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0] 
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # ==========================================================================
    # --- Bản đồ Folium (Nền OpenStreetMap) ---
    # ==========================================================================
    m = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=6, tiles="OpenStreetMap")

    # Che mờ bên ngoài lãnh thổ VN
    world_box = box(-180, -90, 180, 90)
    outside_vn = world_box.difference(vietnam_union)
    folium.GeoJson(outside_vn, style_function=lambda x: {'fillColor': '#ffffff', 'color': 'none', 'fillOpacity': 0.75}, interactive=False).add_to(m)
    
    # Lớp phủ màu nội suy
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_b64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.8).add_to(m)

    # Viền tỉnh
    tooltip_fields = [actual_col] if actual_col in mask_shape.columns else []
    folium.GeoJson(mask_shape, name="Ranh giới chọn", style_function=lambda x: {'fillColor': 'transparent', 'color': '#333333', 'weight': 1.5},
                   tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=['Khu vực: ']) if tooltip_fields else None).add_to(m)

    m.add_child(cm.StepColormap(colors=[mcolors.to_hex(cmap(norm(v))) for v in (custom_levels[:-1] if custom_levels else np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), num_bins))],
                                vmin=np.nanmin(gv_masked), vmax=np.nanmax(gv_masked), index=custom_levels, caption=title_text))

    # ==========================================================================
    # --- Bản đồ Tĩnh Matplotlib (Sửa Font & Tích hợp Contextily OpenStreetMap) ---
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Vẽ lớp màu nội suy (Bám sát ranh giới)
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower', alpha=0.85)
    
    # Nhúng bản đồ OpenStreetMap làm lớp nền (Thay thế cho xa.shp, song.shp)
    try:
        import contextily as cx
        # Thêm nền OSM. Trục của geopandas đang dùng EPSG:4326 nên phải set crs tương ứng
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.9)
    except ImportError:
        st.warning("⚡ Gợi ý: Hãy mở Terminal/CMD gõ lệnh `pip install contextily` để bản đồ tải xuống hiển thị nền OpenStreetMap siêu đẹp!")

    # Vẽ ranh giới các tỉnh
    mask_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)

    # Gắn nhãn tên tỉnh (Fix lỗi font & tạo hộp nền cho dễ đọc)
    for _, row in mask_shape.iterrows():
        centroid = row.geometry.centroid
        name = str(row[actual_col])
        ax.text(centroid.x, centroid.y, name, fontsize=8, fontweight='bold',
                ha='center', va='center', color='#2b2b2b',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f"Giá trị Nội suy", fontsize=11, fontweight='bold')
    
    # Cắt giới hạn trục hiển thị vừa vặn với vùng
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Kinh độ", fontsize=10)
    ax.set_ylabel("Vĩ độ", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    return m, fig, None

# ==============================================================================
# CẤU TRÚC MAIN APP (GIỮ NGUYÊN)
# ==============================================================================
def main():
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
        
        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Nguồn dữ liệu:", ["WeatherObs", "Gió tự động", "Nội suy linh tinh"])
            if obs_mode == "Nội suy linh tinh":
                title_interpol = st.text_input("Tiêu đề bản đồ:", "BẢN ĐỒ NỘI SUY")
                data_file = st.file_uploader("Upload số liệu (.csv/.xlsx):", type=['csv', 'xlsx'])
                
                cmap_list = plt.colormaps()
                cmap_option = st.selectbox("Thang màu:", cmap_list, index=cmap_list.index('jet') if 'jet' in cmap_list else 0)
                
                threshold_type = st.radio("Chia ngưỡng:", ["Tự động", "Nhập tay"])
                num_bins, custom_levels = 10, None
                if threshold_type == "Tự động":
                    num_bins = st.number_input("Số lượng ngưỡng:", 2, 50, 10)
                else:
                    levels_str = st.text_input("Nhập các ngưỡng (cách nhau bằng dấu phẩy):", "0, 10, 20, 30, 40")
                    try: custom_levels = [float(x.strip()) for x in levels_str.split(',') if x.strip()]
                    except: st.error("Lỗi định dạng số.")
                
                # Quét danh sách tỉnh (Thêm Fallback link online)
                province_list = []
                s_col = "NAME_1"
                path_shp = SHP_DISP_PATH if os.path.exists(SHP_DISP_PATH) else SHP_MASK_PATH
                try:
                    if not os.path.exists(path_shp):
                        url_vn = "https://raw.githubusercontent.com/TungTh/tungth.github.io/master/data/vn-provinces.json"
                        temp_s = gpd.read_file(url_vn)
                    else:
                        temp_s = gpd.read_file(path_shp)
                        
                    s_col = next((c for c in ['TEN_TINH', 'NAME_1', 'Name', 'PROVINCE', 'Tỉnh'] if c in temp_s.columns), temp_s.columns[0])
                    province_list = sorted(temp_s[s_col].dropna().astype(str).unique().tolist())
                except: pass
                
                sel_prov = st.multiselect("Tách chọn vùng (Trống = Toàn VN):", province_list)
                
                custom_bounds_dict = None
                if st.checkbox("✂️ Bật giới hạn ô lưới tải (Kinh Vĩ độ)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        min_lon = st.number_input("Kinh độ Min", value=101.80)
                        min_lat = st.number_input("Vĩ độ Min", value=8.00)
                    with c2:
                        max_lon = st.number_input("Kinh độ Max", value=110.00)
                        max_lat = st.number_input("Vĩ độ Max", value=24.00)
                    custom_bounds_dict = {'minx': min_lon, 'maxx': max_lon, 'miny': min_lat, 'maxy': max_lat}

                btn_run = st.button("🚀 VẼ BẢN ĐỒ CHUẨN", type="primary")

                st.markdown("---")
                if st.button("🔒 Đăng xuất"):
                    st.session_state['logged_in'] = False
                    st.rerun()

        if topic == "Dự báo điểm (KMA)":
            if st.session_state['logged_in'] and st.button("🔒 Đăng xuất"):
                st.session_state['logged_in'] = False
                st.rerun()

        if topic == "Bản đồ Bão":
            storm_opt = st.selectbox("Dữ liệu bão:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])
            dashboard_title = st.text_input("Tiêu đề bảng:", "TIN BÃO KHẨN CẤP" if "Hiện trạng" in storm_opt else "THỐNG KÊ LỊCH SỬ")
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
                            sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s)
                            final_df = df[df['storm_no'].isin(sel)] if len(sel)>0 else df
                        else:
                            years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                            temp = df[df['year'].isin(years)]
                            names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                            final_df = temp[temp['name'].isin(names)]
                    except: pass

    # ==========================================================================
    # KHU VỰC HIỂN THỊ CHÍNH
    # ==========================================================================
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?overlay=satellite&product=satellite&zoom=5&lat=16&lon=114")

    elif topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            with st.form("login"):
                u, p = st.text_input("Tên đăng nhập"), st.text_input("Mật khẩu", type="password")
                if st.form_submit_button("Đăng nhập") and u == "admin" and p == "kttv@2026":
                    st.session_state['logged_in'] = True
                    st.rerun()
        else:
            if obs_mode == "WeatherObs":
                st.markdown(f'<div style="overflow:hidden; height:95vh; position:relative;"><iframe src="{LINK_WEATHEROBS}" style="width:100%; height:1000px; position:absolute; top:-50px; border:none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
            elif obs_mode == "Gió tự động":
                st.markdown(f'<div style="overflow:hidden; height:95vh; position:relative;"><iframe src="{LINK_WIND_AUTO}" style="width:100%; height:1200px; position:absolute; top:-75px; border:none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
            elif obs_mode == "Nội suy linh tinh" and btn_run and data_file:
                df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
                with st.spinner("Đang tính toán mảng và nhúng nền OpenStreetMap..."):
                    m_obj, f_obj, err = run_interactive_folium_interpolation(df, title_interpol, cmap_option, num_bins, custom_levels, sel_prov, s_col, custom_bounds_dict)
                if err: st.error(err)
                else:
                    st_folium(m_obj, width=None, height=750, use_container_width=True)
                    buf = io.BytesIO()
                    f_obj.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("⬇️ Tải Bản đồ Xuất bản (Có sẵn nền OpenStreetMap)", buf.getvalue(), "BanDo_NoiSuy_OSM.png", "image/png")

    elif topic == "Dự báo điểm (KMA)":
        if st.session_state['logged_in']:
            st.markdown(f'<div style="overflow:hidden; height:700px; position:relative;"><iframe src="{get_kma_url()}" style="width:100%; height:1200px; position:absolute; top:-130px; border:none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)

    elif topic == "Bản đồ Bão":
        m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
        folium.TileLayer('CartoDB positron', name='Sáng', overlay=False).add_to(m)
        folium.TileLayer('OpenStreetMap', name='Chi tiết', overlay=False).add_to(m)
        ts = get_rainviewer_ts()
        if ts: folium.TileLayer(tiles=f"https://tile.rainviewer.com/{ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png", attr="RainViewer", name="☁️ Mây", overlay=True, show=True, opacity=0.5).add_to(m)

        fg_storm = folium.FeatureGroup(name="🌀 Bão")
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
                        if icon_path and os.path.exists(icon_path):
                            folium.Marker([r['lat'], r['lon']], icon=folium.CustomIcon(image_to_base64(icon_path), icon_size=(40,40)), tooltip=f"Vmax {int(r.get('wind_km/h', 0))}").add_to(fg_storm)
            else:
                for n in final_df['name'].unique():
                    sub = final_df[final_df['name']==n].sort_values('dt')
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg_storm)
                    for _, r in sub.iterrows(): folium.CircleMarker([r['lat'],r['lon']], radius=3, color='#00f2ff' if r.get('wind_km/h',0)<64 else '#ff0055', fill=True, popup=n).add_to(fg_storm)
        
        fg_storm.add_to(m)
        folium.LayerControl(position='topleft').add_to(m)
        
        if show_widgets:
            html_to_render = '<div class="floating-container">'
            if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG): html_to_render += f'<div class="legend-box"><img src="{image_to_base64(CHUTHICH_IMG)}"></div>'
            html_to_render += create_info_table(final_df, dashboard_title) if not final_df.empty else create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU...")
            html_to_render += '</div>'
            st.markdown(html_to_render, unsafe_allow_html=True)
        
        st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
