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
# 3. HÀM XỬ LÝ LOGIC (Rút gọn các hàm phụ trợ đã có sẵn)
# ==============================================================================
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

def run_interactive_folium_interpolation(input_df, title_text, cmap_name, num_bins, custom_levels, selected_provinces, shape_col, custom_bounds=None, mask_gdf=None):
    input_df.columns = input_df.columns.str.lower().str.strip()
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, None, None, "Dữ liệu trống."

    if mask_gdf is not None:
        mask_shape = mask_gdf
    else:
        mask_shape = gpd.read_file(SHP_MASK_PATH)
        if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)

    if custom_bounds:
        minx, maxx, miny, maxy = custom_bounds['minx'], custom_bounds['maxx'], custom_bounds['miny'], custom_bounds['maxy']
        mask_shape = gpd.clip(mask_shape, box(minx, miny, maxx, maxy))
    else:
        minx, miny, maxx, maxy = mask_shape.total_bounds
    
    # Clip dữ liệu điểm nằm trong ranh giới
    pts_geom = [Point(x, y) for x, y in zip(valid['lon'], valid['lat'])]
    valid_gdf = gpd.GeoDataFrame(valid, geometry=pts_geom, crs="EPSG:4326")
    valid = gpd.sjoin(valid_gdf, mask_shape, how="inner").drop(columns='index_right')
    if valid.empty: return None, None, None, "Không có dữ liệu điểm nào trong ranh giới."

    GRID_N, SIGMA = 800, 1.0
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(valid['lon'].values, valid['lat'].values, valid['value'].values, grid_xy, k=12, power=3.0).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    shape_union = mask_shape.unary_union
    prep_shape = prep(shape_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    cmap = plt.get_cmap(cmap_name)
    if custom_levels:
        norm = BoundaryNorm(sorted(list(set(custom_levels))), ncolors=cmap.N, extend='both')
    else:
        v_min, v_max = np.nanmin(gv_masked), np.nanmax(gv_masked)
        custom_levels = np.linspace(v_min if not np.isnan(v_min) else 0, v_max if not np.isnan(v_max) else 1, num_bins + 1)
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')

    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]
    buf = io.BytesIO(); plt.imsave(buf, np.flipud(rgba), format='png'); img_base64 = base64.b64encode(buf.getvalue()).decode()
    
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], tiles="CartoDB positron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_base64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.75).add_to(m)
    folium.GeoJson(mask_shape, style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1.2}).add_to(m)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower')
    mask_shape.boundary.plot(ax=ax, color='black', linewidth=1.0)
    
    cache = {'gv': gv, 'gx': gx, 'gy': gy, 'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy, 'cmap': cmap, 'norm': norm, 'custom_levels': custom_levels, 'mask_shape': mask_shape}
    return m, fig, cache, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    for key in ['folium_map_obj', 'folium_fig_obj', 'interp_cache', 'logged_in']:
        if key not in st.session_state: st.session_state[key] = False if key == 'logged_in' else None

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        obs_mode = ""
        btn_run_interpol = False
        title_interpol = ""
        data_file_interpol = None
        nc_var_selected = None
        nc_time_idx = None

        if topic == "Dữ liệu quan trắc" and st.session_state['logged_in']:
            obs_mode = st.radio("Chọn nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy linh tinh", "Đề tài Quảng Ninh"])
            
            if obs_mode == "Đề tài Quảng Ninh":
                st.markdown("---")
                st.markdown("### 🏛️ ĐỀ TÀI QUẢNG NINH")
                title_interpol = st.text_input("Tiêu đề bản đồ:", value="Bản đồ Quảng Ninh")
                data_file_interpol = st.file_uploader("Upload file (.csv, .xlsx, .nc):", type=['xlsx', 'csv', 'nc'], key="up_qn")
                
                # Xử lý xem trước file NetCDF cho Quảng Ninh
                if data_file_interpol and data_file_interpol.name.endswith('.nc'):
                    try:
                        with open("temp_qn.nc", "wb") as f: f.write(data_file_interpol.getvalue())
                        ds_tmp = xr.open_dataset("temp_qn.nc")
                        vars_list = list(ds_tmp.data_vars.keys())
                        nc_var_selected = st.selectbox("📌 Chọn biến:", vars_list)
                        time_dim = next((d for d in ds_tmp.dims if d.lower() in ['time', 't', 'valid_time']), None)
                        if time_dim:
                            time_vals = [str(t) for t in ds_tmp[time_dim].values]
                            sel_time = st.selectbox("⏳ Bước thời gian:", time_vals)
                            nc_time_idx = time_vals.index(sel_time)
                        ds_tmp.close()
                    except Exception as e: st.error(f"Lỗi đọc NetCDF: {e}")

                cmap_option = st.selectbox("Thang màu:", plt.colormaps(), index=plt.colormaps().index('jet'))
                threshold_type = st.radio("Ngưỡng:", ["Số lớp", "Tùy chỉnh"])
                num_bins, custom_levels = 10, None
                if threshold_type == "Số lớp": num_bins = st.number_input("Số lớp:", 2, 50, 10)
                else:
                    cl_str = st.text_input("Ngưỡng (vd: 0,10,20...):", "0,10,20,30,40,50")
                    custom_levels = [float(x.strip()) for x in cl_str.split(',')]

                btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ QUẢNG NINH", type="primary", use_container_width=True)

        if st.session_state['logged_in']:
            if st.button("🔒 Đăng xuất"): st.session_state['logged_in'] = False; st.rerun()

    # --- MAIN CONTENT ---
    if topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập")
            with st.form("login"):
                u = st.text_input("Tên đăng nhập"); p = st.text_input("Mật khẩu", type="password")
                if st.form_submit_button("Đăng nhập"):
                    if u == "admin" and p == "kttv@2026": st.session_state['logged_in'] = True; st.rerun()
                    else: st.error("Sai thông tin!")
        else:
            if obs_mode == "Đề tài Quảng Ninh":
                if btn_run_interpol and data_file_interpol:
                    try:
                        df_in = pd.DataFrame()
                        # Xử lý trích xuất dữ liệu NetCDF
                        if data_file_interpol.name.endswith('.nc'):
                            with open("run_qn.nc", "wb") as f: f.write(data_file_interpol.getvalue())
                            ds = xr.open_dataset("run_qn.nc")
                            time_dim = next((d for d in ds.dims if d.lower() in ['time', 't', 'valid_time']), None)
                            if time_dim and nc_time_idx is not None: ds = ds.isel({time_dim: nc_time_idx})
                            
                            var_name = nc_var_selected if nc_var_selected else list(ds.data_vars.keys())[0]
                            df_nc = ds[var_name].to_dataframe().reset_index()
                            lat_col = next((c for c in df_nc.columns if c.lower() in ['lat', 'latitude', 'y']), None)
                            lon_col = next((c for c in df_nc.columns if c.lower() in ['lon', 'longitude', 'x']), None)
                            if lat_col and lon_col:
                                df_in = df_nc.rename(columns={lat_col: 'lat', lon_col: 'lon', var_name: 'value'})[['lon', 'lat', 'value']].dropna()
                            ds.close()
                        else:
                            df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)
                        
                        # Xử lý Geodatabase và Vẽ
                        if not df_in.empty and os.path.exists(QN_GDB_PATH):
                            with st.spinner("Đang xử lý dữ liệu Quảng Ninh..."):
                                qn_gdf = gpd.read_file(QN_GDB_PATH)
                                if qn_gdf.crs and qn_gdf.crs.to_epsg() != 4326: qn_gdf = qn_gdf.to_crs(epsg=4326)
                                
                                qn_bounds = {'minx': 106.4, 'maxx': 108.1, 'miny': 20.5, 'maxy': 21.7}
                                m_map, m_fig, m_cache, err = run_interactive_folium_interpolation(
                                    df_in, title_interpol, cmap_option, num_bins, custom_levels, 
                                    None, None, qn_bounds, qn_gdf
                                )
                                if err: st.error(err)
                                else:
                                    st.session_state['folium_map_obj'] = m_map
                                    st.session_state['folium_fig_obj'] = m_fig
                        else: st.error("Không có dữ liệu hoặc thiếu file nen.gdb")
                    except Exception as e: st.error(f"Lỗi hệ thống: {e}")

                if st.session_state['folium_map_obj']:
                    st_folium(st.session_state['folium_map_obj'], width=None, height=800, use_container_width=True)
                    buf = io.BytesIO(); st.session_state['folium_fig_obj'].savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("⬇️ Tải bản đồ Quảng Ninh (PNG)", buf.getvalue(), "quang_ninh.png", "image/png")

            # Các mục khác (WeatherObs, KTTV) giữ nguyên logic iframe của bạn...
            elif obs_mode == "Thời tiết (WeatherObs)":
                 st.markdown(f'<iframe src="{LINK_WEATHEROBS}" style="width:100%; height:95vh;"></iframe>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
