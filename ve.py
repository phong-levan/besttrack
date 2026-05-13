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

# CẬP NHẬT ĐƯỜNG DẪN CHÍNH XÁC THEO GITHUB CỦA BẠN
QN_GDB_PATH = os.path.join("shp", "nen.gdb") 

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
        width: {SIDEBAR_WIDTH} !important; background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd;
    }}
    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; }}
    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HÀM XỬ LÝ LOGIC
# ==============================================================================
def process_nc_to_df(uploaded_file, var_name, time_idx):
    """Hàm xử lý file .nc sang DataFrame (lon, lat, value)"""
    with open("temp_file.nc", "wb") as f:
        f.write(uploaded_file.getvalue())
    ds = xr.open_dataset("temp_file.nc")
    
    time_dim = next((d for d in ds.dims if d.lower() in ['time', 't', 'valid_time']), None)
    if time_dim and time_idx is not None:
        ds_slice = ds.isel({time_dim: time_idx})
    else:
        ds_slice = ds
        
    df = ds_slice[var_name].to_dataframe().reset_index()
    lat_col = next((c for c in df.columns if c.lower() in ['lat', 'latitude', 'y']), None)
    lon_col = next((c for c in df.columns if c.lower() in ['lon', 'longitude', 'x']), None)
    
    if lat_col and lon_col:
        df = df.rename(columns={lat_col: 'lat', lon_col: 'lon', var_name: 'value'})
        df = df[['lon', 'lat', 'value']].dropna()
    
    ds.close()
    if os.path.exists("temp_file.nc"): os.remove("temp_file.nc")
    return df

def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, custom_bounds=None, mask_gdf=None):
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

    if custom_bounds:
        minx, maxx, miny, maxy = custom_bounds['minx'], custom_bounds['maxx'], custom_bounds['miny'], custom_bounds['maxy']
        mask_shape = gpd.clip(mask_shape, box(minx, miny, maxx, maxy))
    else:
        minx, miny, maxx, maxy = mask_shape.total_bounds
    
    # Lọc điểm nằm trong mask
    pts_geom = [Point(x, y) for x, y in zip(valid['lon'], valid['lat'])]
    valid_gdf = gpd.GeoDataFrame(valid, geometry=pts_geom, crs="EPSG:4326")
    valid = gpd.sjoin(valid_gdf, mask_shape, how="inner").drop(columns='index_right')
    if valid.empty: return None, None, None, "Không có dữ liệu điểm nào trong ranh giới."

    # Nội suy IDW
    GRID_N, SIGMA = 800, 1.0
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([valid['lon'].values, valid['lat'].values]))
    dists, idxs = tree.query(grid_xy, k=12)
    w = 1.0 / np.maximum(dists, 1e-12)**3
    gv = (w * valid['value'].values[idxs]).sum(axis=1) / w.sum(axis=1)
    gv = gv.reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    # Cắt theo hình dáng tỉnh
    prep_shape = prep(mask_shape.unary_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    cmap = plt.get_cmap(cmap_name)
    if custom_levels:
        norm = BoundaryNorm(sorted(list(set(custom_levels))), ncolors=cmap.N, extend='both')
    else:
        v_min, v_max = np.nanmin(gv_masked), np.nanmax(gv_masked)
        custom_levels = np.linspace(v_min if not np.isnan(v_min) else 0, v_max if not np.isnan(v_max) else 1, num_bins + 1)
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')

    # Xuất ảnh PNG cho Folium
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], tiles="CartoDB positron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_b64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.8).add_to(m)
    folium.GeoJson(mask_shape, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1.2}).add_to(m)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title_text)
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower')
    mask_shape.boundary.plot(ax=ax, color='black', linewidth=0.8)
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    return m, fig, None, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'f_map' not in st.session_state: st.session_state['f_map'] = None
    if 'f_fig' not in st.session_state: st.session_state['f_fig'] = None

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        obs_mode = ""
        btn_run = False
        data_file = None
        nc_var = None
        nc_time = None

        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy linh tinh", "Đề tài Quảng Ninh"])
            
            if obs_mode in ["Nội suy linh tinh", "Đề tài Quảng Ninh"]:
                title_in = st.text_input("Tiêu đề:", value="Bản đồ nội suy")
                data_file = st.file_uploader("Upload file (.csv, .xlsx, .nc):", type=['csv', 'xlsx', 'nc'])
                
                if data_file and data_file.name.endswith('.nc'):
                    try:
                        with open("temp_meta.nc", "wb") as f: f.write(data_file.getvalue())
                        ds_meta = xr.open_dataset("temp_meta.nc")
                        nc_var = st.selectbox("📌 Biến:", list(ds_meta.data_vars.keys()))
                        time_dim = next((d for d in ds_meta.dims if d.lower() in ['time', 't', 'valid_time']), None)
                        if time_dim:
                            time_opts = [str(t) for t in ds_meta[time_dim].values]
                            nc_time = time_opts.index(st.selectbox("⏳ Thời gian:", time_opts))
                        ds_meta.close()
                    except: st.error("Lỗi đọc file .nc")

                cmap_sel = st.selectbox("Màu:", plt.colormaps(), index=plt.colormaps().index('jet'))
                num_bins = st.number_input("Số lớp:", 2, 50, 10)
                btn_run = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)

        if st.session_state['logged_in']:
            if st.button("🔒 Đăng xuất"): st.session_state['logged_in'] = False; st.rerun()

    if topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập")
            with st.form("login"):
                u = st.text_input("User"); p = st.text_input("Pass", type="password")
                if st.form_submit_button("Vào"):
                    if u == "admin" and p == "kttv@2026": st.session_state['logged_in'] = True; st.rerun()
                    else: st.error("Sai pass!")
        else:
            if obs_mode in ["Nội suy linh tinh", "Đề tài Quảng Ninh"]:
                if btn_run and data_file:
                    try:
                        # 1. Trích xuất dữ liệu
                        if data_file.name.endswith('.nc'):
                            df_pts = process_nc_to_df(data_file, nc_var, nc_time)
                        else:
                            df_pts = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
                        
                        # 2. Xử lý ranh giới
                        mask_g = None
                        bounds = None
                        if obs_mode == "Đề tài Quảng Ninh":
                            if os.path.exists(QN_GDB_PATH):
                                mask_g = gpd.read_file(QN_GDB_PATH)
                                if mask_g.crs and mask_g.crs.to_epsg() != 4326: mask_g = mask_g.to_crs(epsg=4326)
                                bounds = {'minx': 106.4, 'maxx': 108.1, 'miny': 20.5, 'maxy': 21.7}
                            else: st.error(f"⚠️ Không tìm thấy file: {QN_GDB_PATH}")

                        # 3. Vẽ
                        m, fig, _, err = run_interactive_folium_interpolation(df_pts, title_in, cmap_sel, num_bins, None, bounds, mask_g)
                        if err: st.error(err)
                        else:
                            st.session_state['f_map'] = m
                            st.session_state['f_fig'] = fig
                    except Exception as e: st.error(f"Lỗi: {e}")
                
                if st.session_state['f_map']:
                    st_folium(st.session_state['f_map'], width=None, height=800, use_container_width=True)
                    buf = io.BytesIO(); st.session_state['f_fig'].savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("⬇️ Tải ảnh", buf.getvalue(), "map.png", "image/png")

            elif "WeatherObs" in obs_mode:
                st.markdown(f'<iframe src="{LINK_WEATHEROBS}"></iframe>', unsafe_allow_html=True)
            elif "Gió tự động" in obs_mode:
                st.markdown(f'<iframe src="{LINK_WIND_AUTO}"></iframe>', unsafe_allow_html=True)
    
    elif topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?overlay=satellite&zoom=5&lat=16&lon=114", height=1000)

if __name__ == "__main__":
    main()
