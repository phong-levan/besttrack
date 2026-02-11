# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
import requests
from requests.auth import HTTPBasicAuth
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
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU
# ==============================================================================
ICON_DIR = "icon"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# --- CẤU HÌNH ĐƯỜNG DẪN SHAPEFILE ---
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

# --- LINK WEB ---
LINK_WEATHEROBS = "https://weatherobs.com/"
LINK_WIND_AUTO = "https://kttvtudong.net/kttv"

# --- HÀM TẠO LINK KMA DYNAMIC ---
def get_kma_url():
    now_utc = datetime.utcnow()
    check_time = now_utc - timedelta(hours=5)
    run_hour = 0 if check_time.hour < 12 else 12
    date_str = check_time.strftime("%Y.%m.%d")
    tm_str = f"{date_str}.{run_hour:02d}"
    return f"https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm={tm_str}&delta=000&ftm={tm_str}"

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống giám sát",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CSS CHUNG (FIX LỖI MENU & IFRAME)
# ==============================================================================
st.markdown("""
    <style>
    /* Reset padding */
    .block-container { 
        padding: 0 !important; margin: 0 !important; max-width: 100% !important;
    }
    
    /* Header trong suốt, vẫn click được nút menu */
    header { 
        background-color: rgba(0,0,0,0) !important;
        visibility: visible !important;
        z-index: 1000000 !important;
    }
    div[data-testid="stDecoration"] { display: none; }

    /* Sidebar cố định, nền trắng */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        background-color: #ffffff !important;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        border-right: 1px solid #ddd;
    }
    
    /* Màu chữ Sidebar */
    section[data-testid="stSidebar"] * { color: #333333 !important; }

    /* Iframe full màn hình */
    iframe { width: 100% !important; height: 100vh !important; border: none !important; display: block !important; }

    /* Widget nổi */
    .floating-container {
        position: fixed; top: 60px; right: 20px; z-index: 9999;
        display: flex; flex-direction: column; align-items: flex-end;     
    }
    .legend-box { width: 300px; pointer-events: none; margin-bottom: 5px; }
    .info-box {
        width: fit-content; background: rgba(255, 255, 255, 0.95);
        border: 1px solid #ccc; border-radius: 8px;
        padding: 10px !important; color: #000; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
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
    return f"data:image/png;base64,{encoded}"

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename = {
        "tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no",
        "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "thời gian (giờ)": "hour_explicit", 
        "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", 
        "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "khí áp": "pressure", "khí áp (mb)": "pressure", "pmin": "pressure"
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

def create_storm_swaths(dense_df):
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
    status = 'daqua' if 'quá khứ' in str(row.get('status_raw','')).lower() else 'dubao'
    if pd.isna(wind_speed) or wind_speed < 6: return f"vungthap_{status}"
    if wind_speed < 8: return f"atnd_{status}"
    if wind_speed <= 11: return f"bnd_{status}"
    return f"sieubao_{status}"

def create_info_table(df, title):
    if df.empty: return ""
    subtitle = "(Dữ liệu đang cập nhật)"
    rows = ""
    for _, r in df.head(8).iterrows():
        t = r.get('datetime_str', r.get('dt'))
        if not isinstance(t, str): t = t.strftime('%d/%m %Hh')
        bf = r.get('bf', 0)
        rows += f"<tr><td>{t}</td><td>{r.get('lon',0):.1f}E</td><td>{r.get('lat',0):.1f}N</td><td>{f'Cấp {int(bf)}' if bf>0 else '-'}</td><td>{f'{int(r.get('pressure',0))}' if r.get('pressure',0)>0 else '-'}</td></tr>"
    return textwrap.dedent(f"""<div class="info-box"><div style="font-weight:bold;color:red;">{title}</div><div style="font-size:0.8em;">{subtitle}</div><table><thead><tr><th>Giờ</th><th>Kinh</th><th>Vĩ</th><th>Gió</th><th>Pmin</th></tr></thead><tbody>{rows}</tbody></table></div>""")

def create_legend(img_b64):
    return f'<div class="legend-box"><img src="data:image/png;base64,{img_b64}"></div>' if img_b64 else ""

# === LOGIC NỘI SUY ===
def run_interpolation_and_plot(input_df, title_text, data_type='temp'):
    minx, maxx, miny, maxy = 101.8, 115.0, 8.0, 23.9
    GRID_N = 800 # Giảm nhẹ lưới cho nhanh
    
    input_df.columns = input_df.columns.str.lower().str.strip()
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, "Dữ liệu trống."

    if data_type == 'rain':
        vmin, vmax = 0, 1400
        colors = ['#FFFFFF', '#A0E6FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800080', '#4B0082']
        cmap = LinearSegmentedColormap.from_list('rain', colors)
        unit = "Lượng mưa (mm)"
    else:
        vmin, vmax = 0.0, 40.0
        colors = [(0.0, '#FFFFFF'), (0.2, '#00A0FF'), (0.5, '#FFFF00'), (0.8, '#FF0000'), (1.0, '#8B0000')]
        cmap = LinearSegmentedColormap.from_list("temp", colors)
        unit = "Nhiệt độ (°C)"

    x, y, z = valid['lon'].values, valid['lat'].values, valid['value'].values
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    
    # Simple IDW
    tree = cKDTree(np.column_stack([x, y]))
    d, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=10)
    w = 1.0 / np.maximum(d, 1e-12)**3
    gv = (np.sum(w * z[idx], axis=1) / np.sum(w, axis=1)).reshape(gx.shape)
    gv = gaussian_filter(gv, sigma=1)

    # Masking
    try:
        mask = gpd.read_file(SHP_MASK_PATH) if os.path.exists(SHP_MASK_PATH) else gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]})
        if mask.crs and mask.crs.to_epsg() != 4326: mask.to_crs(epsg=4326, inplace=True)
        mask_poly = prep(mask.unary_union)
        mask_flat = [mask_poly.contains(Point(px, py)) for px, py in zip(gx.ravel(), gy.ravel())]
        gv = np.where(np.array(mask_flat).reshape(gx.shape), gv, np.nan)
    except: pass

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title_text, fontsize=15)
    im = ax.imshow(gv, extent=[minx, maxx, miny, maxy], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(im, ax=ax, label=unit, shrink=0.7)
    
    if os.path.exists(SHP_DISP_PATH):
        gpd.read_file(SHP_DISP_PATH).to_crs(epsg=4326).boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        
    return fig, None

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    if 'interpol_fig' not in st.session_state: st.session_state['interpol_fig'] = None
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

    with st.sidebar:
        st.title("Hệ Thống Giám Sát")
        topic = st.radio("CHỨC NĂNG:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""
        obs_mode = ""

        if topic == "Dữ liệu quan trắc":
            if st.session_state['logged_in']:
                obs_options = [
                    "1. Thời tiết (WeatherObs)",
                    "2. Gió trạm KTTV",
                    "3. Gió tự động (kttvtudong)",
                    "4. Nội suy Nhiệt độ",
                    "5. Nội suy Lượng mưa"
                ]
                obs_mode = st.radio("Chọn loại dữ liệu:", obs_options)
                
                if "Nội suy" in obs_mode:
                    st.markdown("---")
                    st.caption("CÔNG CỤ VẼ BẢN ĐỒ")
                    title_interpol = st.text_input("Tiêu đề:", value="Bản đồ nội suy")
                    data_file_interpol = st.file_uploader("Upload số liệu:", type=['xlsx', 'csv'])
                    btn_run_interpol = st.button("VẼ BẢN ĐỒ", type="primary", use_container_width=True)
                
                st.markdown("---")
                if st.button("Đăng xuất"):
                    st.session_state['logged_in'] = False
                    st.rerun()

        elif topic == "Bản đồ Bão":
            active_mode = st.selectbox("Dữ liệu:", ["Hiện trạng (Besttrack)", "Lịch sử"])
            if "Hiện trạng" in active_mode:
                dashboard_title = "TIN BÃO KHẨN CẤP"
                show_widgets = st.checkbox("Hiển thị thông tin", value=True)
                f = st.file_uploader("Upload file (.csv/.xlsx)", key="o1")
                if f:
                    try:
                        df = pd.read_csv(f) if f.name.endswith('csv') else pd.read_excel(f)
                        final_df = normalize_columns(df)
                    except: pass
            else:
                dashboard_title = "LỊCH SỬ BÃO"
                show_widgets = st.checkbox("Hiển thị thông tin", value=True)
                f = st.file_uploader("Upload file (.xlsx)", key="o2")
                if f:
                    try:
                        df = pd.read_excel(f)
                        final_df = normalize_columns(df)
                    except: pass

    # ==============================================================================
    # 5. MAIN CONTENT DISPLAY
    # ==============================================================================
    
    # === ẢNH MÂY VỆ TINH ===
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=1000&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1")
    
    # === DỮ LIỆU QUAN TRẮC ===
    elif topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập Hệ thống")
            with st.form("login"):
                u = st.text_input("User")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if u == "admin" and p == "kttv@2026":
                        st.session_state['logged_in'] = True
                        st.rerun()
                    else: st.error("Sai thông tin")
        else:
            # 1. WEATHER OBS
            if "WeatherObs" in obs_mode:
                st.markdown(f'<iframe src="{LINK_WEATHEROBS}" style="width:100%;height:100vh;border:none;"></iframe>', unsafe_allow_html=True)

            # 2. GIÓ TRẠM KTTV (FIXED: PROXY MODE)
            elif "Gió trạm KTTV" in obs_mode:
                # --- PHẦN QUAN TRỌNG NHẤT: FETCH HTML TỪ SERVER ---
                url_gio = "http://222.255.11.82/Modules/Gio/MapWind.aspx"
                try:
                    # Tải nội dung web về bằng Python (Server-side)
                    r = requests.get(url_gio, auth=HTTPBasicAuth('admin', 'ttdl@2021'), timeout=15, verify=False)
                    
                    if r.status_code == 200:
                        # Chèn base tag để trình duyệt biết load ảnh/css từ đâu
                        html_content = r.text
                        base_tag = '<base href="http://222.255.11.82/Modules/Gio/" target="_self">'
                        
                        # Fix đường dẫn tương đối trong ASP.NET
                        fixed_html = html_content.replace('<head>', f'<head>{base_tag}')
                        
                        # Hiển thị
                        components.html(fixed_html, height=1000, scrolling=True)
                    else:
                        st.error(f"Lỗi kết nối server: {r.status_code}")
                except Exception as e:
                    st.error(f"Không thể tải dữ liệu: {e}")

            # 3. GIÓ TỰ ĐỘNG
            elif "Gió tự động" in obs_mode:
                st.markdown(f'<iframe src="{LINK_WIND_AUTO}" style="width:100%;height:100vh;border:none;margin-top:-50px;"></iframe>', unsafe_allow_html=True)
            
            # 4 & 5. NỘI SUY
            elif "Nội suy" in obs_mode:
                if btn_run_interpol and data_file_interpol:
                    with st.spinner("Đang xử lý..."):
                        df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('csv') else pd.read_excel(data_file_interpol)
                        dtype = 'rain' if "Lượng mưa" in obs_mode else 'temp'
                        fig, err = run_interpolation_and_plot(df_in, title_interpol, dtype)
                        if err: st.error(err)
                        else: st.session_state['interpol_fig'] = fig
                
                if st.session_state['interpol_fig']:
                    st.pyplot(st.session_state['interpol_fig'])
                    buf = io.BytesIO()
                    st.session_state['interpol_fig'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    st.download_button("Tải ảnh", buf.getvalue(), "map.png", "image/png")

    # === KMA ===
    elif topic == "Dự báo điểm (KMA)":
        if st.session_state['logged_in']:
            st.markdown(f'<iframe src="{get_kma_url()}" style="width:100%;height:100vh;margin-top:-100px;border:none;"></iframe>', unsafe_allow_html=True)
        else: st.warning("Vui lòng đăng nhập.")

    # === BÃO ===
    elif topic == "Bản đồ Bão":
        m = folium.Map([16.0, 114.0], zoom_start=6, tiles='CartoDB positron')
        ts = get_rainviewer_ts()
        if ts: folium.TileLayer(f"https://tile.rainviewer.com/{ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png", attr="RainViewer", overlay=True).add_to(m)
        
        fg = folium.FeatureGroup(name="Bão")
        if not final_df.empty:
            if "Hiện trạng" in str(active_mode):
                dense = densify_track(final_df)
                f6, f10, fc = create_storm_swaths(dense)
                for g, c in [(f6,'#FFC0CB'), (f10,'#FF6347'), (fc,'#90EE90')]:
                    if g and not g.is_empty: folium.GeoJson(g, style_function=lambda x,c=c: {'fillColor':c, 'color':c, 'weight':0, 'fillOpacity':0.4}).add_to(fg)
                folium.PolyLine(final_df[['lat','lon']].values.tolist(), color='black').add_to(fg)
                for _, r in final_df.iterrows():
                    ik = get_icon_name(r)
                    ib64 = image_to_base64(ICON_PATHS.get(ik))
                    if ib64: folium.Marker([r['lat'], r['lon']], icon=folium.CustomIcon(ib64, icon_size=(30,30))).add_to(fg)
            else:
                for n, g in final_df.groupby('name'):
                    g = g.sort_values('dt')
                    folium.PolyLine(g[['lat','lon']].values.tolist(), color='blue').add_to(fg)
                    for _, r in g.iterrows(): folium.CircleMarker([r['lat'],r['lon']], radius=2, color='red').add_to(fg)
        
        fg.add_to(m)
        st_folium(m, width=None, height=1000, use_container_width=True)
        if show_widgets and not final_df.empty:
            st.markdown(f'<div class="floating-container">{create_info_table(final_df, dashboard_title)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
