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

# CSS
st.markdown(f"""
    <style>
    .block-container {{ padding: 0 !important; margin: 0 !important; max-width: 100% !important; }}
    header, footer {{ display: none !important; }}
    section[data-testid="stSidebar"] {{
        width: {SIDEBAR_WIDTH} !important; background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd;
    }}
    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; }}
    .floating-container {{ position: fixed; top: 20px; right: 60px; z-index: 9999; display: flex; flex-direction: column; align-items: center; }}
    .info-box {{ background: rgba(255, 255, 255, 0.9); border: 1px solid #ccc; border-radius: 6px; padding: 5px; color: #000; }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HÀM XỬ LÝ LOGIC (GIỮ NGUYÊN CÁC HÀM CƠ BẢN)
# ==============================================================================
@st.cache_data(ttl=300) 
def get_rainviewer_ts():
    try:
        r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=3, verify=False)
        return r.json()['satellite']['infrared'][-1]['time']
    except: return None

def image_to_base64(image_path):
    if not os.path.exists(image_path): return None
    with open(image_path, "rb") as f: encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {"tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no", "thời điểm": "status_raw", "ngày - giờ": "datetime_str", "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h", "cường độ (cấp bf)": "bf"}
    return df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

def idw_knn(xi, yi, zi, query_xy, k=12, power=3.0, eps=1e-12):
    tree = cKDTree(np.column_stack([xi, yi]))
    dists, idxs = tree.query(query_xy, k=min(k, xi.size))
    if dists.ndim == 1: dists, idxs = dists[:, None], idxs[:, None]
    w = 1.0 / np.maximum(dists, eps)**power
    return (w * zi[idxs]).sum(axis=1) / w.sum(axis=1)

def generate_single_province_fig(cache, prov_name, title_text):
    mask_shape = cache.get('mask_shape')
    shape_col = cache.get('shape_col', "")
    if mask_shape is None or not shape_col: return None
    
    prov_shape = mask_shape[mask_shape[shape_col] == prov_name]
    if prov_shape.empty: return None

    p_minx, p_miny, p_maxx, p_maxy = prov_shape.total_bounds
    pad = 0.1
    p_minx -= pad; p_maxx += pad; p_miny -= pad; p_maxy += pad
    
    grid_xy = np.column_stack([cache['gx'].ravel(), cache['gy'].ravel()])
    prep_shape = prep(prov_shape.unary_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(cache['gx'].shape)
    gv_masked = np.where(mask_flat, cache['gv'], np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{title_text}\n(Khu vực: {prov_name})", fontsize=16)
    prov_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)
    im = ax.imshow(gv_masked, extent=[cache['minx'], cache['maxx'], cache['miny'], cache['maxy']], cmap=cache['cmap'], norm=cache['norm'], interpolation='bilinear', origin='lower')
    ax.set_xlim(p_minx, p_maxx); ax.set_ylim(p_miny, p_maxy)
    plt.colorbar(im, ax=ax, extend='both', shrink=0.7)
    return fig

def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None):
    if not os.path.exists(SHP_MASK_PATH): return None, None, None, "Thiếu file vn34tinh.shp"
    
    mask_shape = gpd.read_file(SHP_MASK_PATH)
    if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)
    
    if selected_provinces and shape_col:
        mask_shape = mask_shape[mask_shape[shape_col].isin(selected_provinces)]

    if custom_bounds:
        minx, maxx, miny, maxy = custom_bounds['minx'], custom_bounds['maxx'], custom_bounds['miny'], custom_bounds['maxy']
        mask_shape = gpd.clip(mask_shape, box(minx, miny, maxx, maxy))
    else:
        minx, miny, maxx, maxy = mask_shape.total_bounds

    # Nội suy
    GRID_N = 800
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(input_df['lon'].values, input_df['lat'].values, input_df['value'].values, grid_xy).reshape(gx.shape)
    gv = gaussian_filter(gv, sigma=1.0)

    # Masking
    prep_shape = prep(mask_shape.unary_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    # Thang màu
    cmap = plt.get_cmap(cmap_name)
    if custom_levels: 
        levels = sorted(list(set(custom_levels)))
    else:
        levels = np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), num_bins + 1)
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')

    # Tạo Folium Map
    m = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], tiles="CartoDB positron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Chuyển ảnh nội suy sang base64 để đè lên map
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    img_b64 = base64.b64encode(buf.read()).decode()
    folium.raster_layers.ImageOverlay(f"data:image/png;base64,{img_b64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.7).add_to(m)

    # Thêm ranh giới tỉnh vào map để click
    folium.GeoJson(mask_shape, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1},
                   tooltip=folium.GeoJsonTooltip(fields=[shape_col]) if shape_col else None).add_to(m)

    # Tạo Fig tĩnh để tải
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_shape.boundary.plot(ax=ax, color='black', lw=0.5)
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(im, ax=ax)

    cache = {'gv': gv, 'gx': gx, 'gy': gy, 'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy, 'cmap': cmap, 'norm': norm, 'mask_shape': mask_shape, 'shape_col': shape_col}
    return m, fig, cache, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    for key in ['interpol_fig', 'folium_map_obj', 'folium_fig_obj', 'interp_cache', 'logged_in']:
        if key not in st.session_state: st.session_state[key] = False

    with st.sidebar:
        st.title("Hệ thống Kỹ thuật")
        topic = st.radio("CHẾ ĐỘ:", ["Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)", "Bản đồ Bão"])
        
        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Nguồn:", ["Nội suy linh tinh", "WeatherObs", "Gió tự động"])
            if obs_mode == "Nội suy linh tinh":
                st.subheader("🛠️ Cấu hình Nội suy")
                title_interpol = st.text_input("Tiêu đề:", value="Bản đồ Nội Suy", key="title_custom_interp")
                data_file = st.file_uploader("Upload Data (lon, lat, value):", type=['xlsx', 'csv'])
                
                # --- PHẦN SỬA LỖI SHAPEFILE COLUMN ---
                shape_col = ""
                province_list = []
                if os.path.exists(SHP_MASK_PATH):
                    tmp_shp = gpd.read_file(SHP_MASK_PATH)
                    all_cols = tmp_shp.columns.tolist()
                    potential = ['TEN_TINH', 'NAME_1', 'Name', 'PROVINCE', 'Tỉnh', 'Tinh', 'TENTINH']
                    found_col = next((c for c in potential if c in all_cols), None)
                    
                    st.caption("Cấu hình ranh giới hành chính:")
                    shape_col = st.selectbox("Cột Tên Tỉnh trong SHP:", [found_col] + [c for c in all_cols if c != found_col] if found_col else all_cols)
                    if shape_col:
                        province_list = sorted(tmp_shp[shape_col].dropna().unique().tolist())
                
                selected_provinces = st.multiselect("Lọc theo tỉnh (để trống nếu vẽ hết):", province_list)
                cmap_option = st.selectbox("Thang màu:", plt.colormaps(), index=plt.colormaps().index('jet'))
                
                btn_run = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)

    # Xử lý hiển thị
    if topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            with st.form("login"):
                u, p = st.text_input("User"), st.text_input("Pass", type="password")
                if st.form_submit_button("Login") and u == "admin" and p == "kttv@2026":
                    st.session_state['logged_in'] = True
                    st.rerun()
        else:
            if obs_mode == "Nội suy linh tinh":
                if btn_run and data_file:
                    df_in = pd.read_csv(data_file) if data_file.name.endswith('csv') else pd.read_excel(data_file)
                    m, fig, cache, err = run_interactive_folium_interpolation(df_in, title_interpol, cmap_option, 10, None, selected_provinces, shape_col)
                    if err: st.error(err)
                    else:
                        st.session_state['folium_map_obj'] = m
                        st.session_state['folium_fig_obj'] = fig
                        st.session_state['interp_cache'] = cache

                if st.session_state['folium_map_obj']:
                    map_data = st_folium(st.session_state['folium_map_obj'], width=None, height=700, use_container_width=True)
                    
                    # TẢI TOÀN BỘ
                    buf = io.BytesIO()
                    st.session_state['folium_fig_obj'].savefig(buf, format='png', dpi=300)
                    st.download_button("⬇️ Tải bản đồ tổng (PNG)", buf.getvalue(), "map_tong.png", "image/png")

                    # TẢI TỪNG TỈNH
                    cache = st.session_state['interp_cache']
                    if cache and cache['shape_col']:
                        st.markdown("---")
                        st.subheader("🎯 Trích xuất bản đồ Tỉnh")
                        
                        # Lấy tỉnh từ click hoặc selectbox
                        clicked_prov = None
                        if map_data.get("last_active_drawing"):
                            clicked_prov = map_data["last_active_drawing"].get("properties", {}).get(cache['shape_col'])
                        
                        avail = sorted(cache['mask_shape'][cache['shape_col']].unique())
                        sel_prov = st.selectbox("Chọn tỉnh để trích xuất:", ["-- Chọn --"] + avail, index=avail.index(clicked_prov)+1 if clicked_prov in avail else 0)
                        
                        if sel_prov != "-- Chọn --":
                            p_fig = generate_single_province_fig(cache, sel_prov, title_interpol)
                            if p_fig:
                                buf_p = io.BytesIO()
                                p_fig.savefig(buf_p, format='png', dpi=300)
                                st.pyplot(p_fig)
                                st.download_button(f"⬇️ Tải bản đồ {sel_prov}", buf_p.getvalue(), f"{sel_prov}.png")

    elif topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16&lon=114&zoom=5&overlay=satellite", height=800)

if __name__ == "__main__":
    main()
