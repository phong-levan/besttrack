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

COLOR_SIDEBAR = "#f8f9fa"
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
        width: {SIDEBAR_WIDTH} !important; min-width: {SIDEBAR_WIDTH} !important;
        background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd;
    }}
    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; }}
    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; }}
    .floating-container {{ position: fixed; top: 20px; right: 60px; z-index: 9999; display: flex; flex-direction: column; align-items: center; }}
    .legend-box {{ width: 300px; margin-bottom: 5px; }}
    .info-box {{ background: rgba(255, 255, 255, 0.9); border: 1px solid #ccc; border-radius: 6px; padding: 5px; color: #000; text-align: center; }}
    .info-box table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .info-title {{ font-weight: bold; font-size: 14px; }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HÀM XỬ LÝ LOGIC
# ==============================================================================

def idw_knn(xi, yi, zi, query_xy, k=12, power=3.0):
    tree = cKDTree(np.column_stack([xi, yi]))
    dists, idxs = tree.query(query_xy, k=min(k, xi.size))
    if dists.ndim == 1: dists, idxs = dists[:, None], idxs[:, None]
    w = 1.0 / np.maximum(dists, 1e-12)**power
    return (w * zi[idxs]).sum(axis=1) / w.sum(axis=1)

def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None):
    input_df.columns = input_df.columns.str.lower().str.strip()
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, None, "Dữ liệu trống."

    # Ưu tiên lấy ranh giới từ vungmoi.shp nếu tồn tại, không thì vn34tinh.shp
    path_to_use = SHP_DISP_PATH if os.path.exists(SHP_DISP_PATH) else SHP_MASK_PATH
    if not os.path.exists(path_to_use): return None, None, "Không tìm thấy file ranh giới."
    
    mask_shape = gpd.read_file(path_to_use)
    if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)
        
    # Xác định ranh giới hiển thị (Toàn bộ VN hoặc Tỉnh chọn lọc)
    if selected_provinces:
        # Tự động tìm cột chứa tên tỉnh
        actual_col = next((c for c in mask_shape.columns if any(p in mask_shape[c].astype(str).values for p in selected_provinces)), None)
        if actual_col:
            mask_shape = mask_shape[mask_shape[actual_col].isin(selected_provinces)]
            shape_col = actual_col

    # Tạo ranh giới hợp nhất của Việt Nam để Masking
    vietnam_union = mask_shape.unary_union
    minx, miny, maxx, maxy = (custom_bounds['minx'], custom_bounds['miny'], custom_bounds['maxx'], custom_bounds['maxy']) if custom_bounds else vietnam_union.bounds

    # Tính toán lưới
    GRID_N = 800
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    gv = idw_knn(valid['lon'].values, valid['lat'].values, valid['value'].values, grid_xy).reshape(gx.shape)
    gv = gaussian_filter(gv, sigma=1.0)

    # Cắt mảng theo đúng ranh giới Việt Nam (Clipping)
    prep_shape = prep(vietnam_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    # Thang màu
    cmap = plt.get_cmap(cmap_name)
    if custom_levels:
        norm = BoundaryNorm(sorted(list(set(custom_levels))), ncolors=cmap.N, extend='both')
    else:
        norm = BoundaryNorm(np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), num_bins + 1), ncolors=cmap.N, extend='both')

    # Tạo ảnh RGBA
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0] 
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Tạo Folium Map với nền OpenStreetMap/CartoDB
    m = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=6, tiles="CartoDB positron")

    # Hiệu ứng mờ bên ngoài biên giới VN
    world_box = box(-180, -90, 180, 90)
    outside_vn = world_box.difference(vietnam_union)
    folium.GeoJson(outside_vn, style_function=lambda x: {'fillColor': '#ffffff', 'color': 'none', 'fillOpacity': 0.7}, interactive=False).add_to(m)

    # Đè lớp nội suy lên (Chỉ hiện trong VN)
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_b64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.8).add_to(m)

    # Ranh giới tỉnh tương tác
    folium.GeoJson(mask_shape, name="Ranh giới", style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1},
                   highlight_function=lambda x: {'weight': 2, 'color': 'red', 'fillOpacity': 0.1},
                   tooltip=folium.GeoJsonTooltip(fields=[shape_col], aliases=['Đơn vị: ']) if shape_col in mask_shape.columns else None).add_to(m)

    # Thanh chú giải
    m.add_child(cm.StepColormap(colors=[mcolors.to_hex(cmap(norm(v))) for v in (custom_levels[:-1] if custom_levels else np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), num_bins))],
                                vmin=np.nanmin(gv_masked), vmax=np.nanmax(gv_masked), index=custom_levels, caption=title_text))
    
    # Figure tĩnh để tải về
    fig, ax = plt.subplots(figsize=(10, 12))
    mask_shape.boundary.plot(ax=ax, color='black', linewidth=0.5)
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(title_text)

    return m, fig, None

# (Các hàm bổ trợ khác giữ nguyên như code ban đầu của anh...)
def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {"tên bão": "name", "biển đông": "storm_no", "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h", "cường độ (cấp bf)": "bf"}
    return df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

def main():
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    
    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Nguồn dữ liệu:", ["WeatherObs", "Gió tự động", "Nội suy linh tinh"])
            if obs_mode == "Nội suy linh tinh":
                title_interpol = st.text_input("Tiêu đề:", "Bản đồ Nội suy VN")
                data_file = st.file_uploader("Dữ liệu (.csv/.xlsx):", type=['csv', 'xlsx'])
                cmap_option = st.selectbox("Màu sắc:", plt.colormaps(), index=plt.colormaps().index('jet'))
                
                # Tự động lấy danh sách tỉnh từ file ranh giới
                province_list = []
                s_col = "NAME_1"
                path_shp = SHP_DISP_PATH if os.path.exists(SHP_DISP_PATH) else SHP_MASK_PATH
                if os.path.exists(path_shp):
                    try:
                        temp_s = gpd.read_file(path_shp)
                        s_col = next((c for c in ['TEN_TINH', 'NAME_1', 'Name'] if c in temp_s.columns), temp_s.columns[0])
                        province_list = sorted(temp_s[s_col].dropna().unique().tolist())
                    except: pass
                
                sel_prov = st.multiselect("Chọn vùng (Trống = Toàn VN):", province_list)
                btn_run = st.button("🚀 VẼ BẢN ĐỒ")

    # Phần xử lý hiển thị chính
    if topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            with st.form("login"):
                u, p = st.text_input("User"), st.text_input("Pass", type="password")
                if st.form_submit_button("Đăng nhập") and u == "admin" and p == "kttv@2026":
                    st.session_state['logged_in'] = True
                    st.rerun()
        else:
            if obs_mode == "Nội suy linh tinh" and btn_run and data_file:
                df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
                m_obj, f_obj, err = run_interactive_folium_interpolation(df, title_interpol, cmap_option, 10, None, sel_prov, s_col)
                if err: st.error(err)
                else:
                    st_folium(m_obj, width=None, height=800, use_container_width=True)
                    buf = io.BytesIO()
                    f_obj.savefig(buf, format="png", dpi=300)
                    st.download_button("⬇️ Tải ảnh PNG", buf.getvalue(), "map.png", "image/png")

    elif topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?overlay=satellite&product=satellite&zoom=5&lat=16&lon=114")

if __name__ == "__main__":
    main()
