# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import plugins
from streamlit_folium import st_folium
import os
import base64
import requests
import warnings
import textwrap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.prepared import prep
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import io
from datetime import datetime, timedelta
import xarray as xr

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & TỌA ĐỘ PHẠM VI
# ==============================================================================
SHP_MASK_PATH = os.path.join("shp", "vn34tinh.shp")
GDB_NEN_PATH = os.path.join("shp", "nen.gdb")
GDB_CHUYENDE_PATH = os.path.join("shp", "chuyende.gdb")

# FIX PHẠM VI QUẢNG NINH THEO YÊU CẦU
QN_MINX, QN_MAXX = 106.4, 108.1
QN_MINY, QN_MAXY = 20.5, 21.7

SIDEBAR_WIDTH = "300px"

st.set_page_config(
    page_title="Hệ thống giám sát Quảng Ninh",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. HÀM XỬ LÝ LOGIC NỘI SUY & DỮ LIỆU
# ==============================================================================
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

def get_lat_lon_columns(df_nc):
    lat_candidates = ['lat', 'latitude', 'y', 'lat_0', 'nav_lat']
    lon_candidates = ['lon', 'longitude', 'x', 'lon_0', 'nav_lon']
    lat_col = next((c for c in df_nc.columns if str(c).lower().strip() in lat_candidates or 'lat' in str(c).lower()), None)
    lon_col = next((c for c in df_nc.columns if str(c).lower().strip() in lon_candidates or 'lon' in str(c).lower()), None)
    return lat_col, lon_col

# ==============================================================================
# 3. XỬ LÝ BẢN ĐỒ PHONG CÁCH GIS (THU PHÓNG ĐƯỢC)
# ==============================================================================
def add_gdb_layers_to_folium(m, gdb_path, is_chuyende=False):
    """Đọc và ép phong cách GIS từ các lớp trong file GDB"""
    if not os.path.exists(gdb_path): return
    try:
        import fiona
        layers = fiona.listlayers(gdb_path)
        bbox = box(QN_MINX, QN_MINY, QN_MAXX, QN_MAXY)
        for layer in layers:
            gdf = gpd.read_file(gdb_path, layer=layer)
            if gdf.empty: continue
            if gdf.crs and gdf.crs.to_epsg() != 4326: gdf.to_crs(epsg=4326, inplace=True)
            gdf = gpd.clip(gdf, bbox) # Cắt lớp theo đúng phạm vi fix
            if gdf.empty: continue

            lname = layer.lower()
            if is_chuyende:
                color, weight, fill = '#ff00ff', 1.8, 'transparent' # Màu chuyên đề
            else:
                # Định nghĩa màu sắc chuẩn GIS theo tên layer
                if any(k in lname for k in ['thuy', 'song', 'nuoc']): color, weight, fill = '#00aaff', 1.0, '#cbe7f5'
                elif any(k in lname for k in ['giao', 'duong']): color, weight, fill = '#333333', 0.8, 'transparent'
                elif any(k in lname for k in ['dia', 'dongmuc', 'contour']): color, weight, fill = '#ffd700', 0.5, 'transparent'
                elif any(k in lname for k in ['ranh', 'hanhchinh']): color, weight, fill = '#800080', 1.5, 'transparent'
                else: color, weight, fill = '#888888', 0.5, 'transparent'

            folium.GeoJson(gdf, name=f"Lớp: {layer}", style_function=lambda x, c=color, w=weight, f=fill: {
                'fillColor': f, 'color': c, 'weight': w, 'fillOpacity': 0.7 if f != 'transparent' else 0
            }).add_to(m)
    except: pass

def get_base_gis_map(show_chuyende):
    """Tạo bản đồ nền GIS Quảng Ninh thu phóng được"""
    # Khởi tạo Map với nền biển xanh
    m = folium.Map(location=[(QN_MINY + QN_MAXY)/2, (QN_MINX + QN_MAXX)/2], zoom_start=10, tiles=None)
    m.get_root().html.add_child(folium.Element("<style>.leaflet-container { background: #cbe7f5; }</style>"))
    
    # Lưới tọa độ
    plugins.Graticule(color="#87b0d9", weight=1, opacity=0.8).add_to(m)
    m.fit_bounds([[QN_MINY, QN_MINX], [QN_MAXY, QN_MAXX]])
    
    # Nền đất liền (Trắng)
    if os.path.exists(SHP_MASK_PATH):
        try:
            vn = gpd.read_file(SHP_MASK_PATH)
            folium.GeoJson(vn, name="Đất liền", style_function=lambda x: {
                'fillColor': '#ffffff', 'color': '#cccccc', 'fillOpacity': 1.0, 'weight': 0.5
            }).add_to(m)
        except: pass
    
    # Load các lớp từ GDB
    add_gdb_layers_to_folium(m, GDB_NEN_PATH, is_chuyende=False)
    if show_chuyende:
        add_gdb_layers_to_folium(m, GDB_CHUYENDE_PATH, is_chuyende=True)
    
    folium.LayerControl().add_to(m)
    return m

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (MAIN)
# ==============================================================================
def main():
    if 'logged_in_role' not in st.session_state: st.session_state['logged_in_role'] = None

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)", "Đề Tài Quảng Ninh"])
        
        if topic == "Đề Tài Quảng Ninh" and st.session_state['logged_in_role'] == 'quangninh':
            st.markdown("---")
            st.markdown("### 🛠️ CẤU HÌNH NỘI SUY")
            qn_title = st.text_input("Tiêu đề bản đồ:", value="Bản đồ nội suy Quảng Ninh")
            qn_file = st.file_uploader("Dữ liệu nguồn (.nc, .xlsx, .csv):", type=['nc', 'xlsx', 'csv'])
            qn_cmap = st.selectbox("Thang màu:", plt.colormaps(), index=plt.colormaps().index('jet'))
            qn_bins = st.number_input("Số lớp chia:", 2, 50, 10)
            qn_chuyende = st.checkbox("Hiển thị lớp chuyên đề", value=True)
            btn_run = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)
            if st.button("🔒 Đăng xuất", use_container_width=True):
                st.session_state['logged_in_role'] = None
                st.rerun()

    if topic == "Đề Tài Quảng Ninh":
        if st.session_state['logged_in_role'] != 'quangninh':
            st.title("🔐 Đăng nhập Đề tài Quảng Ninh")
            with st.form("login_qn"):
                u = st.text_input("Tên đăng nhập")
                p = st.text_input("Mật khẩu", type="password")
                if st.form_submit_button("Đăng nhập"):
                    if u == "quangninh2026" and p == "p310":
                        st.session_state['logged_in_role'] = 'quangninh'
                        st.rerun()
                    else: st.error("Tài khoản hoặc mật khẩu không chính xác")
        else:
            if btn_run and qn_file:
                with st.spinner("Đang xử lý dữ liệu và vẽ bản đồ GIS..."):
                    try:
                        # 1. Đọc dữ liệu
                        if qn_file.name.endswith('.nc'):
                            ds = xr.open_dataset(qn_file)
                            var_name = list(ds.data_vars.keys())[0]
                            df_nc = ds[var_name].to_dataframe().reset_index()
                            la, lo = get_lat_lon_columns(df_nc)
                            df_in = df_nc.rename(columns={la: 'lat', lo: 'lon', var_name: 'value'})
                        else:
                            df_in = pd.read_excel(qn_file) if qn_file.name.endswith('xlsx') else pd.read_csv(qn_file)
                        
                        # 2. Tính toán nội suy
                        x_pts, y_pts, z_pts = df_in['lon'].to_numpy(), df_in['lat'].to_numpy(), df_in['value'].to_numpy()
                        gx, gy = np.meshgrid(np.linspace(QN_MINX, QN_MAXX, 800), np.linspace(QN_MINY, QN_MAXY, 800))
                        grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
                        gv = idw_knn(x_pts, y_pts, z_pts, grid_xy).reshape(gx.shape)
                        gv = gaussian_filter(gv, sigma=1.0)
                        
                        # 3. Tạo Overlay ảnh nội suy
                        cmap = plt.get_cmap(qn_cmap)
                        norm = Normalize(vmin=np.nanmin(gv), vmax=np.nanmax(gv))
                        rgba = cmap(norm(gv))
                        rgba[np.isnan(gv)] = [0, 0, 0, 0] # Trong suốt vùng Nan
                        
                        img_buf = io.BytesIO()
                        plt.imsave(img_buf, np.flipud(rgba), format='png')
                        
                        # 4. Hiển thị lên Map GIS động
                        m = get_base_gis_map(qn_chuyende)
                        folium.raster_layers.ImageOverlay(
                            image=f"data:image/png;base64,{base64.b64encode(img_buf.getvalue()).decode()}",
                            bounds=[[QN_MINY, QN_MINX], [QN_MAXY, QN_MAXX]],
                            opacity=0.7,
                            name=qn_title
                        ).add_to(m)
                        
                        st.success("Tạo bản đồ thành công!")
                        st_folium(m, width=None, height=750, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Lỗi xử lý: {str(e)}")
            else:
                # Trạng thái mặc định: Hiển thị bản đồ nền GIS Quảng Ninh
                st.info("👈 Bản đồ nền GIS Quảng Ninh. Vui lòng tải dữ liệu và nhấn 'VẼ BẢN ĐỒ' để xem kết quả.")
                m_base = get_base_gis_map(qn_chuyende if 'qn_chuyende' in locals() else True)
                st_folium(m_base, width=None, height=750, use_container_width=True)

    else:
        st.title("Hệ thống giám sát")
        st.write("Vui lòng chọn các chế độ khác ở thanh menu bên trái.")

if __name__ == "__main__":
    main()
