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
import xarray as xr

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU CỐ ĐỊNH
# ==============================================================================
ICON_DIR = "icon"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")
SHP_MASK_PATH = os.path.join("shp", "vn34tinh.shp")
SHP_DISP_PATH = os.path.join("shp", "vungmoi.shp")
QN_GDB_PATH = os.path.join("shp", "nen.gdb") # Đường dẫn chuẩn theo GitHub của bạn

LINK_WEATHEROBS = "https://weatherobs.com/"
LINK_WIND_AUTO = "https://kttvtudong.net/kttv"

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

# ==============================================================================
# 2. CSS GIAO DIỆN
# ==============================================================================
st.markdown(f"""
    <style>
    .block-container {{ padding: 0 !important; margin: 0 !important; max-width: 100% !important; }}
    header, footer {{ display: none !important; }}
    section[data-testid="stSidebar"] {{
        width: {SIDEBAR_WIDTH} !important; background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd;
    }}
    [data-testid="stAppViewContainer"] {{ padding-left: {SIDEBAR_WIDTH} !important; }}
    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; }}
    .floating-container {{ position: fixed; top: 20px; right: 60px; z-index: 9999; display: flex; flex-direction: column; align-items: center; }}
    .info-box {{
        width: fit-content; background: rgba(255, 255, 255, 0.9); border: 1px solid #ccc; border-radius: 6px;
        padding: 8px; color: #000; text-align: center; font-size: 12px;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HÀM BỔ TRỢ & XỬ LÝ DỮ LIỆU
# ==============================================================================
def process_nc_to_df(uploaded_file, var_name, time_idx):
    with open("temp_file.nc", "wb") as f:
        f.write(uploaded_file.getvalue())
    ds = xr.open_dataset("temp_file.nc")
    time_dim = next((d for d in ds.dims if d.lower() in ['time', 't', 'valid_time']), None)
    ds_slice = ds.isel({time_dim: time_idx}) if time_dim and time_idx is not None else ds
    df = ds_slice[var_name].to_dataframe().reset_index()
    lat_col = next((c for c in df.columns if c.lower() in ['lat', 'latitude', 'y']), None)
    lon_col = next((c for c in df.columns if c.lower() in ['lon', 'longitude', 'x']), None)
    if lat_col and lon_col:
        df = df.rename(columns={lat_col: 'lat', lon_col: 'lon', var_name: 'value'})
        df = df[['lon', 'lat', 'value']].dropna()
    ds.close()
    return df

def run_interpolation_logic(df, title, cmap_name, bins, levels, bounds=None, mask_gdf=None):
    df.columns = df.columns.str.lower().str.strip()
    valid = df.dropna(subset=['lon', 'lat', 'value']).copy()
    
    if mask_gdf is not None:
        mask_shape = mask_gdf
    else:
        mask_shape = gpd.read_file(SHP_MASK_PATH) if os.path.exists(SHP_MASK_PATH) else None
        if mask_shape is not None and mask_shape.crs and mask_shape.crs.to_epsg() != 4326:
            mask_shape = mask_shape.to_crs(epsg=4326)

    if bounds:
        minx, maxx, miny, maxy = bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy']
        if mask_shape is not None: mask_shape = gpd.clip(mask_shape, box(minx, miny, maxx, maxy))
    else:
        minx, miny, maxx, maxy = mask_shape.total_bounds if mask_shape is not None else (102, 8, 110, 24)

    # Lọc điểm trong ranh giới
    pts_geom = [Point(x, y) for x, y in zip(valid['lon'], valid['lat'])]
    valid_gdf = gpd.GeoDataFrame(valid, geometry=pts_geom, crs="EPSG:4326")
    if mask_shape is not None:
        valid = gpd.sjoin(valid_gdf, mask_shape, how="inner").drop(columns='index_right')

    # IDW
    GRID_N = 800
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    tree = cKDTree(np.column_stack([valid['lon'].values, valid['lat'].values]))
    dists, idxs = tree.query(grid_xy, k=12)
    w = 1.0 / np.maximum(dists, 1e-12)**3
    gv = (w * valid['value'].values[idxs]).sum(axis=1) / w.sum(axis=1)
    gv = gaussian_filter(gv.reshape(gx.shape), sigma=1.0)

    # Masking
    if mask_shape is not None:
        prep_shape = prep(mask_shape.unary_union)
        mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
        gv_masked = np.where(mask_flat, gv, np.nan)
    else: gv_masked = gv

    cmap = plt.get_cmap(cmap_name)
    norm = BoundaryNorm(levels if levels else np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), bins+1), ncolors=cmap.N, extend='both')

    # Folium
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], tiles="CartoDB positron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_b64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.8).add_to(m)
    if mask_shape is not None:
        folium.GeoJson(mask_shape, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1.2}).add_to(m)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower')
    if mask_shape is not None: mask_shape.boundary.plot(ax=ax, color='black', linewidth=0.8)
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    return m, fig

# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==============================================================================
def main():
    # Khởi tạo session state
    for key in ['logged_in', 'f_map', 'f_fig', 'storm_df']:
        if key not in st.session_state: st.session_state[key] = False if key == 'logged_in' else None

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        obs_mode = ""
        btn_run = False
        data_file = None
        nc_var, nc_time = None, None

        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy linh tinh", "Đề tài Quảng Ninh"])
            
            if obs_mode in ["Nội suy linh tinh", "Đề tài Quảng Ninh"]:
                title_in = st.text_input("Tiêu đề:", value="Bản đồ nội suy")
                data_file = st.file_uploader("Upload file:", type=['csv', 'xlsx', 'nc'])
                
                if data_file and data_file.name.endswith('.nc'):
                    try:
                        with open("temp_sidebar.nc", "wb") as f: f.write(data_file.getvalue())
                        ds = xr.open_dataset("temp_sidebar.nc")
                        nc_var = st.selectbox("Biến:", list(ds.data_vars.keys()))
                        time_dim = next((d for d in ds.dims if d.lower() in ['time', 't']), None)
                        if time_dim:
                            opts = [str(t) for t in ds[time_dim].values]
                            nc_time = opts.index(st.selectbox("Thời gian:", opts))
                        ds.close()
                    except: st.error("Lỗi đọc file .nc")

                cmap_sel = st.selectbox("Màu:", plt.colormaps(), index=plt.colormaps().index('jet'))
                num_bins = st.number_input("Số lớp:", 2, 50, 10)
                btn_run = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)

        if st.session_state['logged_in']:
            if st.button("🔒 Đăng xuất"): st.session_state['logged_in'] = False; st.rerun()

    # --- NỘI DUNG CHÍNH ---
    if topic == "Bản đồ Bão":
        st.subheader("🌀 Theo dõi bão trực tuyến")
        m_storm = folium.Map(location=[16, 114], zoom_start=5, tiles="CartoDB positron")
        st_folium(m_storm, width=None, height=800, use_container_width=True)
        st.info("👈 Sử dụng thanh bên để tải dữ liệu bão (Besttrack).")

    elif topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?overlay=satellite&zoom=5&lat=16&lon=114", height=1000)

    elif topic == "Dự báo điểm (KMA)":
        components.iframe(get_kma_url(), height=800)

    elif topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập")
            with st.form("login"):
                u = st.text_input("User"); p = st.text_input("Pass", type="password")
                if st.form_submit_button("Vào"):
                    if u == "admin" and p == "kttv@2026": 
                        st.session_state['logged_in'] = True
                        st.rerun()
                    else: st.error("Sai tài khoản!")
        else:
            if obs_mode in ["Nội suy linh tinh", "Đề tài Quảng Ninh"]:
                if btn_run and data_file:
                    try:
                        df_pts = process_nc_to_df(data_file, nc_var, nc_time) if data_file.name.endswith('.nc') else (pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file))
                        
                        mask_g, bounds = None, None
                        if obs_mode == "Đề tài Quảng Ninh":
                            if os.path.exists(QN_GDB_PATH):
                                mask_g = gpd.read_file(QN_GDB_PATH)
                                if mask_g.crs and mask_g.crs.to_epsg() != 4326: mask_g = mask_g.to_crs(epsg=4326)
                                bounds = {'minx': 106.4, 'maxx': 108.1, 'miny': 20.5, 'maxy': 21.7}
                            else: st.error(f"⚠️ Thiếu: {QN_GDB_PATH}")

                        m, fig = run_interpolation_logic(df_pts, title_in, cmap_sel, num_bins, None, bounds, mask_g)
                        st.session_state['f_map'], st.session_state['f_fig'] = m, fig
                    except Exception as e: st.error(f"Lỗi: {e}")

                if st.session_state['f_map']:
                    st_folium(st.session_state['f_map'], width=None, height=800, use_container_width=True)
                    buf = io.BytesIO()
                    st.session_state['f_fig'].savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("⬇️ Tải bản đồ", buf.getvalue(), "map.png", "image/png")
                else:
                    st.info("👈 Hãy cấu hình và nhấn 'VẼ BẢN ĐỒ'.")

            elif obs_mode == "Thời tiết (WeatherObs)":
                components.iframe(LINK_WEATHEROBS, height=1000)
            elif obs_mode == "Gió tự động (KTTV)":
                components.iframe(LINK_WIND_AUTO, height=1000)

if __name__ == "__main__":
    main()
