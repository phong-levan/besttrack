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
QN_GDB_PATH = "nen.gdb" 

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

st.set_page_config(page_title="Hệ thống giám sát KTTV", layout="wide", initial_sidebar_state="expanded")

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

def process_nc_to_df(uploaded_file, var_name, time_idx):
    """Trích xuất dữ liệu từ NetCDF sang DataFrame (lon, lat, value)"""
    with open("temp_data.nc", "wb") as f:
        f.write(uploaded_file.getvalue())
    ds = xr.open_dataset("temp_data.nc")
    
    # Lấy slice theo thời gian nếu có
    time_dim = next((d for d in ds.dims if d.lower() in ['time', 't', 'valid_time']), None)
    if time_dim and time_idx is not None:
        ds_slice = ds.isel({time_dim: time_idx})
    else:
        ds_slice = ds
    
    df = ds_slice[var_name].to_dataframe().reset_index()
    # Tìm cột tọa độ
    lat_col = next((c for c in df.columns if c.lower() in ['lat', 'latitude', 'y']), None)
    lon_col = next((c for c in df.columns if c.lower() in ['lon', 'longitude', 'x']), None)
    
    if lat_col and lon_col:
        df = df.rename(columns={lat_col: 'lat', lon_col: 'lon', var_name: 'value'})
        df = df[['lon', 'lat', 'value']].dropna()
    
    ds.close()
    if os.path.exists("temp_data.nc"): os.remove("temp_data.nc")
    return df

def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, custom_bounds=None, mask_gdf=None):
    """Hàm nội suy chính và tạo bản đồ Folium"""
    input_df.columns = input_df.columns.str.lower().str.strip()
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, None, None, "Dữ liệu trống."

    # Xử lý mask ranh giới
    if mask_gdf is not None:
        mask_shape = mask_gdf
    else:
        mask_shape = gpd.read_file(SHP_MASK_PATH) if os.path.exists(SHP_MASK_PATH) else None
        if mask_shape is not None and mask_shape.crs and mask_shape.crs.to_epsg() != 4326:
            mask_shape = mask_shape.to_crs(epsg=4326)

    # Khống chế phạm vi (Cắt clip)
    if custom_bounds:
        minx, maxx, miny, maxy = custom_bounds['minx'], custom_bounds['maxx'], custom_bounds['miny'], custom_bounds['maxy']
        if mask_shape is not None:
            mask_shape = gpd.clip(mask_shape, box(minx, miny, maxx, maxy))
    else:
        minx, miny, maxx, maxy = mask_shape.total_bounds if mask_shape is not None else (102, 8, 110, 24)

    # Chỉ nội suy các điểm nằm trong mask
    pts_geom = [Point(x, y) for x, y in zip(valid['lon'], valid['lat'])]
    valid_gdf = gpd.GeoDataFrame(valid, geometry=pts_geom, crs="EPSG:4326")
    if mask_shape is not None:
        valid = gpd.sjoin(valid_gdf, mask_shape, how="inner").drop(columns='index_right')
    if valid.empty: return None, None, None, "Không có dữ liệu điểm nào trong ranh giới."

    GRID_N, SIGMA = 800, 1.0
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(valid['lon'].values, valid['lat'].values, valid['value'].values, grid_xy, k=12, power=3.0).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    # Tạo mask mảng cho plot
    if mask_shape is not None:
        shape_union = mask_shape.unary_union
        prep_shape = prep(shape_union)
        mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
        gv_masked = np.where(mask_flat, gv, np.nan)
    else:
        gv_masked = gv

    cmap = plt.get_cmap(cmap_name)
    if custom_levels:
        norm = BoundaryNorm(sorted(list(set(custom_levels))), ncolors=cmap.N, extend='both')
    else:
        v_min, v_max = np.nanmin(gv_masked), np.nanmax(gv_masked)
        custom_levels = np.linspace(v_min if not np.isnan(v_min) else 0, v_max if not np.isnan(v_max) else 1, num_bins + 1)
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')

    # Convert to base64 image for Folium
    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], tiles="CartoDB positron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_base64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.8).add_to(m)
    if mask_shape is not None:
        folium.GeoJson(mask_shape, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1.2}).add_to(m)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title_text)
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower')
    if mask_shape is not None: mask_shape.boundary.plot(ax=ax, color='black', linewidth=0.8)
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    return m, fig, None, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'folium_map' not in st.session_state: st.session_state['folium_map'] = None
    if 'folium_fig' not in st.session_state: st.session_state['folium_fig'] = None

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        obs_mode = ""
        btn_run = False
        title_in = ""
        data_file = None
        nc_var = None
        nc_time = None

        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Chọn nguồn:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy linh tinh", "Đề tài Quảng Ninh"])
            
            if obs_mode in ["Nội suy linh tinh", "Đề tài Quảng Ninh"]:
                st.markdown(f"### 🛠️ {obs_mode.upper()}")
                title_in = st.text_input("Tiêu đề bản đồ:", value=f"Bản đồ nội suy {'Quảng Ninh' if 'Quảng' in obs_mode else ''}")
                data_file = st.file_uploader("Upload file (.csv, .xlsx, .nc):", type=['xlsx', 'csv', 'nc'])
                
                # Cấu hình NetCDF nếu file là .nc
                if data_file and data_file.name.endswith('.nc'):
                    try:
                        with open("temp_meta.nc", "wb") as f: f.write(data_file.getvalue())
                        ds_meta = xr.open_dataset("temp_meta.nc")
                        nc_var = st.selectbox("📌 Chọn biến dữ liệu:", list(ds_meta.data_vars.keys()))
                        time_dim = next((d for d in ds_meta.dims if d.lower() in ['time', 't', 'valid_time']), None)
                        if time_dim:
                            time_options = [str(t) for t in ds_meta[time_dim].values]
                            sel_time = st.selectbox("⏳ Chọn thời gian:", time_options)
                            nc_time = time_options.index(sel_time)
                        ds_meta.close()
                    except Exception as e: st.error(f"Lỗi đọc file NetCDF: {e}")

                cmap_opt = st.selectbox("Thang màu:", plt.colormaps(), index=plt.colormaps().index('jet'))
                th_type = st.radio("Ngưỡng chia:", ["Tự động", "Tùy chỉnh"])
                n_bins, c_levels = 10, None
                if th_type == "Tự động": n_bins = st.number_input("Số lớp:", 2, 50, 10)
                else:
                    cl_str = st.text_input("Dãy ngưỡng (cách nhau dấu phẩy):", "0,10,20,30,40,50")
                    try: c_levels = [float(x.strip()) for x in cl_str.split(',')]
                    except: pass
                
                btn_run = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)

        if st.session_state['logged_in']:
            if st.button("🔒 Đăng xuất"):
                st.session_state['logged_in'] = False
                st.rerun()

    # --- LOGIC XỬ LÝ CHÍNH ---
    if topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập hệ thống")
            with st.form("login_form"):
                u = st.text_input("Tên đăng nhập"); p = st.text_input("Mật khẩu", type="password")
                if st.form_submit_button("Đăng nhập"):
                    if u == "admin" and p == "kttv@2026":
                        st.session_state['logged_in'] = True; st.rerun()
                    else: st.error("Sai thông tin đăng nhập!")
        else:
            if obs_mode in ["Nội suy linh tinh", "Đề tài Quảng Ninh"]:
                if btn_run and data_file:
                    try:
                        # 1. Đọc dữ liệu (NC hoặc Excel/CSV)
                        if data_file.name.endswith('.nc'):
                            df_pts = process_nc_to_df(data_file, nc_var, nc_time)
                        else:
                            df_pts = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
                        
                        # 2. Xử lý ranh giới & Phạm vi
                        mask_g = None
                        bounds = None
                        if obs_mode == "Đề tài Quảng Ninh":
                            if os.path.exists(QN_GDB_PATH):
                                mask_g = gpd.read_file(QN_GDB_PATH)
                                if mask_g.crs and mask_g.crs.to_epsg() != 4326: mask_g = mask_g.to_crs(epsg=4326)
                                bounds = {'minx': 106.4, 'maxx': 108.1, 'miny': 20.5, 'maxy': 21.7}
                            else: st.error(f"Thiếu file {QN_GDB_PATH}")

                        # 3. Vẽ bản đồ
                        m, fig, _, err = run_interactive_folium_interpolation(df_pts, title_in, cmap_opt, n_bins, c_levels, bounds, mask_g)
                        if err: st.error(err)
                        else:
                            st.session_state['folium_map'] = m
                            st.session_state['folium_fig'] = fig
                    except Exception as e: st.error(f"Lỗi xử lý: {e}")
                
                if st.session_state['folium_map']:
                    st_folium(st.session_state['folium_map'], width=None, height=800, use_container_width=True)
                    buf = io.BytesIO()
                    st.session_state['folium_fig'].savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("⬇️ Tải ảnh bản đồ (PNG)", buf.getvalue(), "map.png", "image/png")

            elif obs_mode == "Thời tiết (WeatherObs)":
                st.markdown(f'<iframe src="{LINK_WEATHEROBS}"></iframe>', unsafe_allow_html=True)
            elif obs_mode == "Gió tự động (KTTV)":
                st.markdown(f'<iframe src="{LINK_WIND_AUTO}"></iframe>', unsafe_allow_html=True)

    elif topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?overlay=satellite&zoom=5&lat=16&lon=114", height=1000)

if __name__ == "__main__":
    main()
