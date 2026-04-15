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
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# --- CẤU HÌNH ĐƯỜNG DẪN SHAPEFILE ---
SHP_MASK_PATH = os.path.join("shp", "vn34tinh.shp")
SHP_DISP_PATH = os.path.join("shp", "vungmoi.shp")

# THÊM CẤU HÌNH ĐƯỜNG DẪN SHAPEFILE XÃ VÀ SÔNG NGÒI
SHP_XA_PATH = os.path.join("shp", "xa.shp")       # Đổi tên file cho khớp với của anh nếu cần
SHP_SONG_PATH = os.path.join("shp", "song.shp")   # Đổi tên file cho khớp với của anh nếu cần

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
        "tên bão": "name", "biển đông": "storm_no", "vĩ độ": "lat", "kinh độ": "lon", 
        "vmax (km/h)": "wind_km/h", "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "khí áp": "pressure", "thời điểm": "status_raw", "ngày - giờ": "datetime_str"
    }
    return df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

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

    path_to_use = SHP_DISP_PATH if os.path.exists(SHP_DISP_PATH) else SHP_MASK_PATH
    if not os.path.exists(path_to_use): return None, None, "Không tìm thấy file ranh giới."
    
    mask_shape = gpd.read_file(path_to_use)
    if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)
        
    if selected_provinces:
        actual_col = next((c for c in mask_shape.columns if any(p in mask_shape[c].astype(str).values for p in selected_provinces)), None)
        if actual_col:
            mask_shape = mask_shape[mask_shape[actual_col].isin(selected_provinces)]
            shape_col = actual_col

    vietnam_union = mask_shape.unary_union
    minx, miny, maxx, maxy = (custom_bounds['minx'], custom_bounds['miny'], custom_bounds['maxx'], custom_bounds['maxy']) if custom_bounds else vietnam_union.bounds

    GRID_N = 800
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    gv = idw_knn(valid['lon'].values, valid['lat'].values, valid['value'].values, grid_xy).reshape(gx.shape)
    gv = gaussian_filter(gv, sigma=1.0)

    prep_shape = prep(vietnam_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
    gv_masked = np.where(mask_flat, gv, np.nan)

    cmap = plt.get_cmap(cmap_name)
    if custom_levels:
        norm = BoundaryNorm(sorted(list(set(custom_levels))), ncolors=cmap.N, extend='both')
    else:
        norm = BoundaryNorm(np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), num_bins + 1), ncolors=cmap.N, extend='both')

    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0] 
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # --- Bản đồ Folium ---
    m = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=6, tiles="CartoDB positron")

    world_box = box(-180, -90, 180, 90)
    outside_vn = world_box.difference(vietnam_union)
    folium.GeoJson(outside_vn, style_function=lambda x: {'fillColor': '#ffffff', 'color': 'none', 'fillOpacity': 0.7}, interactive=False).add_to(m)

    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{img_b64}", bounds=[[miny, minx], [maxy, maxx]], opacity=0.8).add_to(m)

    folium.GeoJson(mask_shape, name="Ranh giới", style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1},
                   highlight_function=lambda x: {'weight': 2, 'color': 'red', 'fillOpacity': 0.1},
                   tooltip=folium.GeoJsonTooltip(fields=[shape_col], aliases=['Đơn vị: ']) if shape_col in mask_shape.columns else None).add_to(m)

    m.add_child(cm.StepColormap(colors=[mcolors.to_hex(cmap(norm(v))) for v in (custom_levels[:-1] if custom_levels else np.linspace(np.nanmin(gv_masked), np.nanmax(gv_masked), num_bins))],
                                vmin=np.nanmin(gv_masked), vmax=np.nanmax(gv_masked), index=custom_levels, caption=title_text))
    
    # ==============================================================================
    # --- Figure tĩnh (Matplotlib) nâng cao (Thêm Xã, Sông ngòi & Legend) ---
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 1. Vẽ Ảnh Nội suy
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, origin='lower')
    
    # Danh sách các thành phần để tạo Bảng chú giải (Legend)
    legend_handles = []

    # 2. Đọc và Vẽ Ranh giới Xã (Cắt theo khu vực làm việc)
    if os.path.exists(SHP_XA_PATH):
        try:
            xa_shp = gpd.read_file(SHP_XA_PATH)
            if xa_shp.crs and xa_shp.crs.to_epsg() != 4326: xa_shp.to_crs(epsg=4326, inplace=True)
            # Clip để bản đồ đỡ nặng và sạch sẽ
            xa_shp_clipped = gpd.clip(xa_shp, mask_shape)
            if not xa_shp_clipped.empty:
                xa_shp_clipped.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.2, alpha=0.7)
                legend_handles.append(mlines.Line2D([], [], color='gray', linewidth=0.5, label='Ranh giới Xã'))
        except Exception as e:
            pass

    # 3. Đọc và Vẽ Sông ngòi (Cắt theo khu vực làm việc)
    if os.path.exists(SHP_SONG_PATH):
        try:
            song_shp = gpd.read_file(SHP_SONG_PATH)
            if song_shp.crs and song_shp.crs.to_epsg() != 4326: song_shp.to_crs(epsg=4326, inplace=True)
            # Clip sông ngòi
            song_shp_clipped = gpd.clip(song_shp, mask_shape)
            if not song_shp_clipped.empty:
                song_shp_clipped.plot(ax=ax, color='blue', linewidth=0.6, alpha=0.8)
                legend_handles.append(mlines.Line2D([], [], color='blue', linewidth=1.0, label='Sông ngòi'))
        except Exception as e:
            pass

    # 4. Vẽ Ranh giới Tỉnh/Vùng (Lớp trên cùng để làm nổi bật)
    mask_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0)
    legend_handles.append(mlines.Line2D([], [], color='black', linewidth=1.5, label='Ranh giới Tỉnh'))

    # 5. Thêm Bảng Chú giải (Legend)
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=10, facecolor='white', framealpha=0.9)

    # 6. Thêm Colorbar nội suy
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    # 7. Tinh chỉnh Layout
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Kinh độ")
    ax.set_ylabel("Vĩ độ")
    ax.ticklabel_format(useOffset=False, style='plain')

    return m, fig, None

def run_interpolation_and_plot(input_df, title_text, data_type='temp'):
    # Hàm tĩnh cũ (Giữ nguyên theo yêu cầu)
    return None, None

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
                    f_obj.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("⬇️ Tải bản đồ Tĩnh (Chuẩn)", buf.getvalue(), "ban_do_chuan.png", "image/png")

    elif topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?overlay=satellite&product=satellite&zoom=5&lat=16&lon=114")

if __name__ == "__main__":
    main()
