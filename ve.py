# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
import folium.plugins
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
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

SHP_MASK_PATH = os.path.join("shp", "vn34tinh.shp")
SHP_DISP_PATH = os.path.join("shp", "vungmoi.shp")
GDB_NEN_PATH = os.path.join("shp", "nen.gdb")
GDB_CHUYENDE_PATH = os.path.join("shp", "chuyende.gdb")

# --- FIX PHẠM VI (BOUNDING BOX) QUẢNG NINH ---
QN_MINX, QN_MAXX = 106.4, 108.1
QN_MINY, QN_MAXY = 20.5, 21.7

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

SIDEBAR_WIDTH = "300px"

st.set_page_config(page_title="Hệ thống giám sát", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
    .block-container {{ padding: 0 !important; margin: 0 !important; max-width: 100% !important; }}
    header, footer {{ display: none !important; }}
    div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{
        visibility: hidden !important; display: none !important; height: 0px !important;
    }}
    section[data-testid="stSidebar"] {{
        width: {SIDEBAR_WIDTH} !important; min-width: {SIDEBAR_WIDTH} !important;
        background-color: #f8f9fa !important; border-right: 1px solid #ddd;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# HÀM XỬ LÝ LOGIC
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
# XỬ LÝ QUẢNG NINH - TỰ ĐỘNG ĐỌC LAYER KHÔNG CẦN FIONA
# ==============================================================================
def add_gdb_to_folium(m, gdb_path, is_chuyende=False):
    if not os.path.exists(gdb_path): return
    try:
        import pyogrio
        layers = pyogrio.list_layers(gdb_path)
        bbox = box(QN_MINX, QN_MINY, QN_MAXX, QN_MAXY)
        for layer_info in layers:
            layer = layer_info[0]
            gdf = gpd.read_file(gdb_path, layer=layer)
            if gdf.empty: continue
            if gdf.crs and gdf.crs.to_epsg() != 4326: gdf.to_crs(epsg=4326, inplace=True)
            gdf = gpd.clip(gdf, bbox)
            if gdf.empty: continue
            
            lname = layer.lower()
            color, weight, fill, fill_op = 'gray', 0.8, 'transparent', 0
            
            if is_chuyende:
                color, weight = '#ff0000', 1.5
            else:
                if gdf.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
                    fill, fill_op, color = '#ffffff', 1.0, '#cccccc'
                    if any(k in lname for k in ['thuy', 'song', 'ho', 'nuoc', 'bien']): fill, color = '#cbe7f5', '#00aaff'
                    elif 'nen' in lname or 'admin' in lname: fill = '#ffffb3'
                else:
                    if any(k in lname for k in ['thuy', 'song', 'ho']): color, weight = '#00aaff', 1.0
                    elif any(k in lname for k in ['giao', 'duong']): color, weight = '#000000', 0.8
                    elif any(k in lname for k in ['dia', 'dongmuc']): color, weight = '#ffd700', 0.5
                    elif any(k in lname for k in ['ranh', 'bien']): color, weight = '#800080', 1.5

            folium.GeoJson(gdf, name=layer, style_function=lambda x,c=color,w=weight,f=fill,fo=fill_op: {'fillColor': f, 'color': c, 'weight': w, 'fillOpacity': fo}).add_to(m)
    except: pass

def get_default_qn_map(show_chuyende):
    m = folium.Map(location=[(QN_MINY+QN_MAXY)/2, (QN_MINX+QN_MAXX)/2], zoom_start=10, tiles=None)
    m.get_root().html.add_child(folium.Element("<style>.leaflet-container { background: #cbe7f5; }</style>"))
    folium.plugins.Graticule(color="#87b0d9", weight=1).add_to(m)
    m.fit_bounds([[QN_MINY, QN_MINX], [QN_MAXY, QN_MAXX]])
    
    if os.path.exists(SHP_MASK_PATH):
        vn = gpd.read_file(SHP_MASK_PATH)
        folium.GeoJson(vn, style_function=lambda x: {'fillColor': '#ffffff', 'color': '#cccccc', 'fillOpacity': 1.0, 'weight': 0.5}).add_to(m)
        
    add_gdb_to_folium(m, GDB_NEN_PATH, False)
    if show_chuyende: add_gdb_to_folium(m, GDB_CHUYENDE_PATH, True)
    folium.LayerControl().add_to(m)
    return m

def run_qn_folium_interpolation(input_df, title_text, cmap_option, num_bins, custom_levels, show_chuyende):
    input_df.columns = input_df.columns.str.lower().str.strip()
    x_pts, y_pts, z_pts = input_df['lon'].to_numpy(), input_df['lat'].to_numpy(), input_df['value'].to_numpy()
    gx, gy = np.meshgrid(np.linspace(QN_MINX, QN_MAXX, 800), np.linspace(QN_MINY, QN_MAXY, 800))
    gv = idw_knn(x_pts, y_pts, z_pts, np.column_stack([gx.ravel(), gy.ravel()]), k=12, power=3.0).reshape(gx.shape)
    gv = gaussian_filter(gv, sigma=1.0)
    
    cmap = plt.get_cmap(cmap_option)
    if custom_levels:
        norm = BoundaryNorm(sorted(custom_levels), ncolors=cmap.N, extend='both')
    else:
        levels = np.linspace(np.nanmin(gv), np.nanmax(gv), num_bins + 1)
        norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')

    rgba = cmap(norm(gv))
    rgba[np.isnan(gv)] = [0, 0, 0, 0]
    
    m = get_default_qn_map(show_chuyende)
    img_overlay = io.BytesIO()
    plt.imsave(img_overlay, np.flipud(rgba), format='png')
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{base64.b64encode(img_overlay.getvalue()).decode()}", bounds=[[QN_MINY, QN_MINX], [QN_MAXY, QN_MAXX]], opacity=0.7).add_to(m)
    
    # Tạo Figure tĩnh để tải về
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#cbe7f5')
    ax.imshow(gv, extent=[QN_MINX, QN_MAXX, QN_MINY, QN_MAXY], cmap=cmap, norm=norm, origin='lower', aspect='auto')
    ax.set_title(title_text)
    
    return m, fig, None

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    if 'logged_in_role' not in st.session_state: st.session_state['logged_in_role'] = None

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)", "Đề Tài Quảng Ninh"])
        
        if topic == "Đề Tài Quảng Ninh" and st.session_state['logged_in_role'] == 'quangninh':
            st.markdown("---")
            qn_title = st.text_input("Tiêu đề:", value="Bản đồ Quảng Ninh")
            qn_file = st.file_uploader("Dữ liệu (.nc/.xlsx/.csv):", type=['nc', 'xlsx', 'csv'])
            qn_cmap = st.selectbox("Màu sắc:", plt.colormaps(), index=plt.colormaps().index('jet'))
            qn_bins = st.number_input("Số lớp:", 2, 50, 10)
            qn_chuyende = st.checkbox("Bật lớp Chuyên đề", value=False)
            btn_run = st.button("🚀 VẼ BẢN ĐỒ")
            if st.button("🔒 Đăng xuất"): st.session_state['logged_in_role'] = None; st.rerun()

    if topic == "Đề Tài Quảng Ninh":
        if st.session_state['logged_in_role'] != 'quangninh':
            with st.form("login_qn"):
                u = st.text_input("User"); p = st.text_input("Pass", type="password")
                if st.form_submit_button("Login"):
                    if u == "quangninh2026" and p == "p310":
                        st.session_state['logged_in_role'] = 'quangninh'; st.rerun()
                    else: st.error("Sai tài khoản")
        else:
            if qn_file and btn_run:
                if qn_file.name.endswith('.nc'):
                    ds = xr.open_dataset(qn_file)
                    var = list(ds.data_vars.keys())[0]
                    df_in = ds[var].to_dataframe().reset_index()
                    la, lo = get_lat_lon_columns(df_in)
                    df_in = df_in.rename(columns={la: 'lat', lo: 'lon', var: 'value'})
                else:
                    df_in = pd.read_excel(qn_file) if qn_file.name.endswith('xlsx') else pd.read_csv(qn_file)
                
                m, fig, _ = run_qn_folium_interpolation(df_in, qn_title, qn_cmap, qn_bins, None, qn_chuyende)
                st_folium(m, width=None, height=750)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300); buf.seek(0)
                st.download_button("⬇️ Tải ảnh tĩnh", buf, "map.png", "image/png")
            else:
                st_folium(get_default_qn_map(qn_chuyende if 'qn_chuyende' in locals() else False), width=None, height=750)

if __name__ == "__main__":
    main()
