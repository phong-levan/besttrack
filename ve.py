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
import geopandas as gpd
from shapely.geometry import Point, box, Polygon, mapping
from shapely.prepared import prep
from shapely.ops import unary_union
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import io

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.csv"
FILE_OPT2 = "besttrack_capgio.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# --- CẤU HÌNH ĐƯỜNG DẪN SHAPEFILE CỐ ĐỊNH (TỪ GIT/THƯ MỤC LOCAL) ---
# Dùng đường dẫn tương đối. Streamlit sẽ tìm trong thư mục 'shp' nằm cùng cấp với file code.
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
# 2. CSS CHUNG (FIX CỨNG SIDEBAR & ĐẨY NỘI DUNG)
# ==============================================================================
st.markdown(f"""
    <style>
    /* 1. THIẾT LẬP CHUNG - XÓA PADDING/MARGIN */
    .block-container {{
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }}
    header, footer {{
        display: none !important;
    }}
    
    div[data-testid="stToolbar"], 
    div[data-testid="stDecoration"], 
    div[data-testid="stStatusWidget"] {{
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }}

    /* 2. ÉP SIDEBAR LUÔN HIỆN CỐ ĐỊNH BÊN TRÁI */
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
    [data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}

    /* 3. ĐẨY NỘI DUNG CHÍNH SANG PHẢI */
    [data-testid="stAppViewContainer"] {{
        padding-left: {SIDEBAR_WIDTH} !important;
        padding-top: 0 !important;
    }}
    [data-testid="stMainViewContainer"] {{
        margin-left: 0 !important;
        width: 100% !important;
        padding-top: 0 !important;
    }}

    /* 4. TỐI ƯU CHO IFRAME */
    iframe {{
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
    }}

    /* 5. WIDGET NỔI (CONTAINER CHỨA CẢ 2) */
    .floating-container {{
        position: fixed; 
        top: 20px; 
        right: 60px; 
        z-index: 9999;
        display: flex;
        flex-direction: column; 
        align-items: center;    
    }}

    /* BẢNG CHÚ THÍCH (LEGEND) */
    .legend-box {{
        width: 340px; 
        pointer-events: none;
        margin-bottom: 5px;
    }}
    
    /* BẢNG THÔNG TIN */
    .info-box {{
        width: fit-content; 
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #ccc; 
        border-radius: 6px;
        padding: 10px !important; 
        color: #000;
        text-align: center;
    }}
    
    .info-box table {{
        width: 100%;
        margin: 0 auto;
        border-collapse: collapse;
    }}
    .info-box th, .info-box td {{
        text-align: center !important; 
        padding: 4px 8px;
    }}
    .info-title {{
        font-weight: bold;
        margin-bottom: 2px;
    }}
    .info-subtitle {{
        font-size: 0.9em;
        margin-bottom: 8px;
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
    if not os.path.exists(image_path):
        return None
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
            elif w < 134: wind_speed = 12
    
    status_raw = str(row.get('status_raw','')).lower()
    
    status = 'dubao' 
    if 'quá khứ' in status_raw or 'past' in status_raw:
        status = 'daqua'
    
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

    subtitle = ""
    try:
        target_row = None
        if 'status_raw' in df.columns:
            current_rows = df[df['status_raw'].astype(str).str.strip().str.lower() == 'hiện tại']
            if not current_rows.empty:
                target_row = current_rows.iloc[0]
            else:
                 current_rows = df[df['status_raw'].astype(str).str.contains("hiện tại|current", case=False, na=False)]
                 if not current_rows.empty:
                    target_row = current_rows.iloc[0]
        
        if target_row is None and not display_df.empty:
            target_row = display_df.iloc[0]

        if target_row is not None:
            if 'hour_explicit' in target_row.index and pd.notna(target_row['hour_explicit']):
                h = int(target_row['hour_explicit'])
                subtitle = f"Tin phát lúc {h}h30"
            elif 'dt' in target_row.index and pd.notna(target_row['dt']):
                subtitle = f"Tin phát lúc {target_row['dt'].hour}h30"
            else:
                 subtitle = "(Đang cập nhật)"
        else:
             subtitle = "(Đang cập nhật)"
    except:
        subtitle = "(Dữ liệu cập nhật từ Besttrack)"
    
    rows = ""
    for _, r in display_df.iterrows():
        t = r.get('datetime_str', r.get('dt'))
        if not isinstance(t, str): t = t.strftime('%d/%m %Hh')
        w = r.get('wind_km/h', 0)
        
        lon = f"{r.get('lon', 0):.1f}E"
        lat = f"{r.get('lat', 0):.1f}N"
        
        bf = r.get('bf', 0)
        if (pd.isna(bf) or bf == 0) and w > 0:
             if w < 34: bf = 6
             elif w < 64: bf = 8
             elif w < 100: bf = 10
             else: bf = 12
        cap_gio = f"Cấp {int(bf)}" if bf > 0 else "-"
        
        p = r.get('pressure', 0)
        pmin = f"{int(p)}" if (pd.notna(p) and p > 0) else "-"

        rows += f"<tr><td>{t}</td><td>{lon}</td><td>{lat}</td><td>{cap_gio}</td><td>{pmin}</td></tr>"
    
    return textwrap.dedent(f"""
    <div class="info-box">
        <div class="info-title">{title}</div>
        <div class="info-subtitle">{subtitle}</div>
        <table>
            <thead>
                <tr>
                    <th>Ngày-Giờ</th>
                    <th>Kinh độ</th>
                    <th>Vĩ độ</th>
                    <th>Cấp gió</th>
                    <th>Pmin</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""")

def create_legend(img_b64):
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="legend-box">
        <img src="data:image/png;base64,{img_b64}">
    </div>""")

# === LOGIC NỘI SUY NHIỆT ĐỘ ===
def idw_knn(xi, yi, zi, query_xy, k=12, power=3.0, eps=1e-12):
    tree = cKDTree(np.column_stack([xi, yi]))
    dists, idxs = tree.query(query_xy, k=min(k, xi.size))
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    exact = dists <= eps
    out = np.empty(dists.shape[0], dtype=float)

    if np.any(exact):
        ex_idx_rows = np.where(exact.any(axis=1))[0]
        for r in ex_idx_rows:
            c = np.where(exact[r])[0][0]
            out[r] = zi[idxs[r, c]]

    rest = ~exact.any(axis=1)
    if np.any(rest):
        d = dists[rest]
        nn = idxs[rest]
        w = 1.0 / np.maximum(d, eps)**power
        z_sel = zi[nn]              
        out[rest] = (w * z_sel).sum(axis=1) / w.sum(axis=1)

    return out

def run_interpolation_and_plot(input_df, title_text):
    # Cấu hình nội suy
    minx, maxx = 101.8, 115.0
    miny, maxy = 8.0, 23.9
    GRID_N = 1000 
    SIGMA = 1.5
    IDW_POWER = 3.0
    KNN = 12
    vmin, vmax = 0.0, 40.0 

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

    gx, gy = np.meshgrid(
        np.linspace(minx, maxx, GRID_N),
        np.linspace(miny, maxy, GRID_N)
    )
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    gv = idw_knn(xi, yi, zi, grid_xy, k=KNN, power=IDW_POWER).reshape(gx.shape)

    if SIGMA > 0:
        gv = gaussian_filter(gv, sigma=SIGMA)

    mask_shape = None
    disp_shape = None
    
    # 1. Đọc Shapefile cố định từ thư mục 'shp' trên Git/Local
    # Kiểm tra sự tồn tại của file trước
    if os.path.exists(SHP_MASK_PATH):
        try:
            # Đọc trực tiếp file .shp
            mask_shape = gpd.read_file(SHP_MASK_PATH)
            
            # Chuẩn hóa hệ tọa độ về WGS84 (EPSG:4326)
            if mask_shape.crs and mask_shape.crs.to_epsg() != 4326:
                mask_shape.to_crs(epsg=4326, inplace=True)
        except Exception as e:
            return None, f"Lỗi đọc file Mask Shapefile ({SHP_MASK_PATH}): {e}"
    else:
        # Nếu không thấy file, dùng khung chữ nhật mặc định và thông báo (nếu cần)
        # st.warning(f"Không tìm thấy file: {SHP_MASK_PATH}. Đang dùng khung mặc định.")
        bbox_poly = box(minx, miny, maxx, maxy)
        mask_shape = gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs='EPSG:4326')

    if os.path.exists(SHP_DISP_PATH):
        try:
            disp_shape = gpd.read_file(SHP_DISP_PATH)
            if disp_shape.crs and disp_shape.crs.to_epsg() != 4326:
                disp_shape.to_crs(epsg=4326, inplace=True)
        except Exception as e:
            return None, f"Lỗi đọc file Display Shapefile ({SHP_DISP_PATH}): {e}"
    else:
        if mask_shape is not None:
             disp_shape = mask_shape # Dùng mask làm display nếu thiếu file display
        else:
             bbox_poly = box(minx, miny, maxx, maxy)
             disp_shape = gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs='EPSG:4326')

    # 2. Cắt Mask
    if mask_shape is not None:
        shape_union = mask_shape.unary_union
        prep_shape = prep(shape_union)
        mask_flat = np.fromiter(
            (prep_shape.contains(Point(px, py)) for px, py in grid_xy),
            count=grid_xy.shape[0],
            dtype=bool
        ).reshape(gx.shape)
        gv_masked = np.where(mask_flat, gv, np.nan)
    else:
        gv_masked = gv

    # 3. Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(14, 10)) 
    ax.set_title(title_text if title_text else 'Bản đồ nhiệt độ', fontsize=16)

    # Vẽ đường viền
    if disp_shape is not None:
        disp_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    colors = [
        (0.0, '#FFFFFF'), (0.1, '#D0F0FF'), (0.2, '#00A0FF'), (0.4, '#00FF00'),
        (0.6, '#FFFF00'), (0.75, '#FFA500'), (0.9, '#FF0000'), (1.0, '#8B0000')
    ]
    cmap = LinearSegmentedColormap.from_list("custom_smooth_temp", colors, N=256)
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels_for_ticks = list(range(0, 42, 4))

    im = ax.imshow(
        gv_masked,
        extent=[minx, maxx, miny, maxy],
        cmap=cmap,
        norm=norm,
        interpolation='bilinear',
        origin='lower'
    )

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, extend='both')
    cbar.set_label('Nhiệt độ (°C)', fontsize=12)
    cbar.set_ticks(levels_for_ticks)
    cbar.set_ticklabels([str(l) for l in levels_for_ticks])

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.ticklabel_format(useOffset=False, style='plain')
    
    return fig, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    
    if 'interpol_fig' not in st.session_state:
        st.session_state['interpol_fig'] = None

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        
        topic = st.radio("CHỌN CHẾ ĐỘ:", 
                        ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""
        
        obs_mode = ""
        
        # --- CÁC BIẾN CHO NỘI SUY (KHỞI TẠO) ---
        title_interpol = ""
        data_file_interpol = None
        btn_run_interpol = False

        if topic == "Dữ liệu quan trắc":
            obs_mode = st.radio("Chọn nguồn dữ liệu:", 
                              ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy nhiệt độ"])
            
            # --- CHUYỂN TOÀN BỘ CÔNG CỤ NỘI SUY VÀO SIDEBAR ---
            if obs_mode == "Nội suy nhiệt độ":
                st.markdown("---")
                st.markdown("### 🛠️ CÔNG CỤ NỘI SUY")
                
                # *** PHẦN NHẬP TIÊU ĐỀ ***
                title_interpol = st.text_input("Tiêu đề bản đồ:", value="Bản đồ nhiệt độ nội suy")
                
                st.markdown("**1. Upload dữ liệu (.xlsx/.csv)**")
                st.caption("Cột: `stations`, `lon`, `lat`, `value`")
                data_file_interpol = st.file_uploader("Chọn file số liệu:", type=['xlsx', 'csv'], key="data_up")
                
                # BỎ PHẦN UPLOAD SHAPEFILE, CODE TỰ TÌM TRONG THƯ MỤC 'shp'
                
                st.markdown("---")
                btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)

        if topic == "Bản đồ Bão":
            storm_opt = st.selectbox("Dữ liệu bão:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])
            active_mode = storm_opt
            
            if "Hiện trạng" in storm_opt:
                dashboard_title = "TIN BÃO KHẨN CẤP"
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack.csv", type="csv", key="o1")
                    path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                    
                    def process_file(f_path):
                        if not f_path: return pd.DataFrame()
                        try:
                            if isinstance(f_path, str):
                                if f_path.endswith('.csv'): df = pd.read_csv(f_path)
                                else: df = pd.read_excel(f_path)
                            else: 
                                if f_path.name.endswith('.csv'): df = pd.read_csv(f_path)
                                else: df = pd.read_excel(f_path)
                                
                            df = normalize_columns(df)
                            if 'name' not in df.columns and 'storm_no' not in df.columns:
                                df['name'] = 'Cơn bão'
                                df['storm_no'] = 'Current Storm'

                            for c in ['wind_km/h', 'bf', 'r6', 'r10', 'rc', 'pressure', 'hour_explicit']: 
                                if c not in df.columns: df[c] = 0
                            if 'datetime_str' in df.columns: df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
                            elif all(c in df.columns for c in ['year','mon','day','hour']): df['dt'] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day, hour=df.hour), errors='coerce')
                            for c in ['lat','lon','wind_km/h', 'pressure', 'bf']: df[c] = pd.to_numeric(df[c], errors='coerce')
                            return df.dropna(subset=['lat','lon'])
                        except: return pd.DataFrame()

                    df = process_file(path)
                    if not df.empty:
                        if 'storm_no' in df.columns:
                            all_s = df['storm_no'].unique()
                            sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s)
                            final_df = df[df['storm_no'].isin(sel)]
                        else:
                            final_df = df
                    else: st.warning("Vui lòng tải file.")
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
                            for c in ['wind_km/h', 'bf', 'r6', 'r10', 'rc', 'pressure']: 
                                if c not in df.columns: df[c] = 0
                            if 'datetime_str' in df.columns: df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
                            elif all(c in df.columns for c in ['year','mon','day','hour']): df['dt'] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day, hour=df.hour), errors='coerce')
                            for c in ['lat','lon','wind_km/h', 'pressure']: df[c] = pd.to_numeric(df[c], errors='coerce')
                            df = df.dropna(subset=['lat','lon'])

                            years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                            temp = df[df['year'].isin(years)]
                            names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                            final_df = temp[temp['name'].isin(names)]
                        except: pass
                    else: st.warning("Vui lòng tải file.")

    # --- MAIN CONTENT ---
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=1000&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1")
    
    elif topic == "Dữ liệu quan trắc":
        
        if "WeatherObs" in obs_mode:
            html_weather = f"""
            <div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;">
                <iframe 
                    src="{LINK_WEATHEROBS}" 
                    style="
                        width: calc(100% + 19px); 
                        height: 1000px; 
                        position: absolute; 
                        top: -65px; 
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
                        top: -100px;    
                        left: 0px; 
                        border: none;"
                    allow="fullscreen"
                ></iframe>
            </div>
            """
             st.markdown(html_kttv, unsafe_allow_html=True)
        
        elif "Nội suy nhiệt độ" in obs_mode:
            # === PHẦN HIỂN THỊ KẾT QUẢ NỘI SUY (MAIN AREA) ===
            
            # Xử lý khi nhấn nút Vẽ
            if btn_run_interpol:
                if data_file_interpol:
                    try:
                        if data_file_interpol.name.endswith('.csv'):
                            df_in = pd.read_csv(data_file_interpol)
                        else:
                            df_in = pd.read_excel(data_file_interpol)
                        
                        with st.spinner("Đang tính toán nội suy và tạo bản đồ..."):
                            # Gọi hàm không cần uploaded_shp nữa
                            fig, err = run_interpolation_and_plot(df_in, title_interpol)
                            
                            if err:
                                st.error(f"❌ {err}")
                            else:
                                st.session_state['interpol_fig'] = fig
                                
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xử lý file: {e}")
                else:
                    st.toast("Vui lòng upload file dữ liệu trước!", icon="⚠️")

            # Hiển thị biểu đồ nếu đã có trong session state
            if st.session_state['interpol_fig']:
                st.pyplot(st.session_state['interpol_fig'], use_container_width=True)
                
                # --- Nút Download ---
                st.markdown("### 📥 Tải xuống")
                col_dl1, col_dl2 = st.columns([1, 3])
                with col_dl1:
                    fmt = st.selectbox("Định dạng:", ["png", "pdf"])
                
                # Lưu vào buffer để download
                buf = io.BytesIO()
                st.session_state['interpol_fig'].savefig(buf, format=fmt, dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                with col_dl2:
                    st.write("") # Spacer
                    st.write("") # Spacer
                    st.download_button(
                        label=f"⬇️ Tải ảnh về ({fmt.upper()})",
                        data=buf,
                        file_name=f"ban_do_nhiet.{fmt}",
                        mime=f"image/{fmt}"
                    )
            else:
                # Màn hình chờ
                st.info("👈 Vui lòng cấu hình và nhấn nút 'VẼ BẢN ĐỒ' ở thanh menu bên trái.")

    elif topic == "Dự báo điểm (KMA)":
        html_kma = f"""
        <div style="overflow: hidden; width: 100%; height: 700px; position: relative; border: 1px solid #ddd;">
            <iframe 
                src="{LINK_KMA_FORECAST}" 
                style="
                    width: calc(100% + 19px); 
                    height: 1200px; 
                    position: absolute; 
                    top: -215px; 
                    left: 0px; 
                    border: none;"
                allow="fullscreen"
            ></iframe>
        </div>
        """
        st.markdown(html_kma, unsafe_allow_html=True)

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
                    
                    # --- VẼ ICON BÃO ---
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
                            folium.Marker(location=[r['lat'], r['lon']], icon=icon, tooltip=f"Gió: {r.get('wind_km/h', 0)} km/h").add_to(fg_storm)
            else: 
                for n in final_df['name'].unique():
                    sub = final_df[final_df['name']==n].sort_values('dt')
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg_storm)
                    for _, r in sub.iterrows():
                        c = '#00f2ff' if r.get('wind_km/h',0)<64 else '#ff0055'
                        folium.CircleMarker([r['lat'],r['lon']], radius=3, color=c, fill=True, popup=f"{n}").add_to(fg_storm)
        
        fg_storm.add_to(m)
        folium.LayerControl(position='topleft', collapsed=False).add_to(m)
        
        # --- HIỂN THỊ WIDGET TRONG CONTAINER CHUNG ---
        if show_widgets:
            html_to_render = '<div class="floating-container">'
            
            # 1. Thêm Chú thích (Nếu có)
            if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG):
                with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                html_to_render += create_legend(b64)
            
            # 2. Thêm Bảng thông tin
            if not final_df.empty: 
                html_to_render += create_info_table(final_df, dashboard_title)
            else: 
                html_to_render += create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU...")
            
            html_to_render += '</div>'
            st.markdown(html_to_render, unsafe_allow_html=True)
        
        st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
