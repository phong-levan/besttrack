# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import re
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FuncFormatter
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

# --- CẤU HÌNH ĐƯỜNG DẪN SHAPEFILE CỐ ĐỊNH ---
SHP_MASK_PATH = os.path.join("shp", "vn34tinh.shp")
SHP_DISP_PATH = os.path.join("shp", "vungmoi.shp")
SHP_XA_PATH   = os.path.join("shp", "RG_xa_VN.shp")      # <-- Lớp ranh giới xã

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

# --- DANH SÁCH LINK WEB ---
LINK_WEATHEROBS = "https://weatherobs.com/"
LINK_WIND_AUTO = "https://kttvtudong.net/kttv"

# --- HÀM TẠO LINK KMA DYNAMIC ---
def get_kma_url():
    now_utc = datetime.utcnow()
    check_time = now_utc - timedelta(hours=5)
    run_hour = 0 if check_time.hour < 12 else 12
    date_str = check_time.strftime("%Y.%m.%d")
    tm_str = f"{date_str}.{run_hour:02d}"
    url = f"https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm={tm_str}&delta=000&ftm={tm_str}"
    return url

COLOR_BG = "#ffffff"
COLOR_SIDEBAR = "#f8f9fa"
COLOR_TEXT = "#333333"
COLOR_ACCENT = "#007bff"
COLOR_BORDER = "#dee2e6"
SIDEBAR_WIDTH = "300px"

st.set_page_config(
    page_title="Hệ thống giám sát",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Trạng thái đóng/mở sidebar do ta tự quản lý (đáng tin cậy hơn dò DOM của Streamlit) ----
if 'sidebar_open' not in st.session_state:
    st.session_state['sidebar_open'] = True
_SB_TRANSFORM = "translateX(0)" if st.session_state['sidebar_open'] else "translateX(-100%)"
_SB_PADDING = SIDEBAR_WIDTH if st.session_state['sidebar_open'] else "0px"

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
        display: block !important; visibility: visible !important;
        width: {SIDEBAR_WIDTH} !important; min-width: {SIDEBAR_WIDTH} !important; max-width: {SIDEBAR_WIDTH} !important;
        position: fixed !important; left: 0 !important; top: 0 !important; height: 100vh !important;
        z-index: 100000 !important; background-color: {COLOR_SIDEBAR} !important; border-right: 1px solid #ddd;
        transform: {_SB_TRANSFORM} !important; transition: transform 0.25s ease;
    }}
    [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseBtn"], [data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    [data-testid="stAppViewContainer"] {{ padding-left: {_SB_PADDING} !important; padding-top: 0 !important; transition: padding-left 0.25s ease; }}
    .st-key-btn_toggle_sidebar_wrap {{
        position: fixed !important; top: 14px !important; left: 14px !important; z-index: 100001 !important;
        width: 42px !important;
    }}
    .st-key-btn_toggle_sidebar_wrap button {{
        width: 42px !important; height: 42px !important; padding: 0 !important; font-size: 18px !important;
        border-radius: 8px !important; border: 1px solid {COLOR_BORDER} !important;
        background: #ffffff !important; box-shadow: 0 1px 4px rgba(0,0,0,0.15) !important;
    }}
    [data-testid="stMainViewContainer"] {{ margin-left: 0 !important; width: 100% !important; padding-top: 0 !important; }}
    iframe {{ width: 100% !important; height: 100vh !important; border: none !important; display: block !important; }}
    .floating-container {{ position: fixed; top: 20px; right: 60px; z-index: 9999; display: flex; flex-direction: column; align-items: center; }}
    .legend-box {{ width: 300px; pointer-events: none; margin-bottom: 5px; }}
    .info-box {{
        width: fit-content; background: rgba(255, 255, 255, 0.9); border: 1px solid #ccc; border-radius: 6px;
        padding: 5px !important; color: #000; text-align: center;
    }}
    .info-box table {{ width: 100%; margin: 0 auto; border-collapse: collapse; }}
    .info-box th, .info-box td {{ text-align: center !important; padding: 2px 5px !important; font-size: 12px !important; }}
    .info-title {{ font-weight: bold; margin-bottom: 2px; font-size: 14px !important; }}
    .info-subtitle {{ font-size: 10px !important; margin-bottom: 5px; font-style: italic; }}
    </style>
""", unsafe_allow_html=True)

# ---- Nút đóng/mở sidebar (ô vuông ☰) - tự quản lý bằng session_state, luôn hiển thị ----
try:
    _toggle_ctx = st.container(key="btn_toggle_sidebar_wrap")
except TypeError:
    _toggle_ctx = st.container()
with _toggle_ctx:
    if st.button("☰", key="btn_toggle_sidebar"):
        st.session_state['sidebar_open'] = not st.session_state['sidebar_open']
        st.rerun()

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
        "tên bão": "name", "biển đông": "storm_no", "số hiệu": "storm_no", "thời điểm": "status_raw", "ngày - giờ": "datetime_str",
        "thời gian (giờ)": "hour_explicit", "vĩ độ": "lat", "kinh độ": "lon", "vmax (km/h)": "wind_km/h",
        "cường độ (cấp bf)": "bf", "bán kính gió mạnh cấp 6 (km)": "r6", "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "khí áp": "pressure", "khí áp (mb)": "pressure", "pmin": "pressure", "pmin (mb)": "pressure"
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    return df

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

def them_nut_chup_anh_ban_do(m, ten_file="ban_do_bao", rong_px=6000, cao_px=4000, ten_tile_layer=None):
    """
    Thêm nút 'chụp ảnh' (📷) lên bản đồ Folium tương tác, dùng plugin leaflet-easyPrint để xuất
    đúng khung hình đang xem (nền bản đồ, đường đi bão, lưới tọa độ...) ra file PNG độ phân giải cao.
    Nút do ta tự vẽ (luôn hiển thị ngay cả khi thư viện chưa tải xong) và đặt ở góc dưới-phải
    để tránh bị che bởi khung Chú Thích/Bảng tin ở góc trên-phải.
    """
    ten_map = m.get_name()
    ten_tile = ten_tile_layer.get_name() if ten_tile_layer is not None else "null"
    script = f"""
    <script>
        (function() {{
            function _taiScript(src, ok, loi) {{
                var s = document.createElement('script');
                s.src = src;
                s.onload = ok;
                s.onerror = loi;
                document.head.appendChild(s);
            }}

            function _khoiTaoNutChup() {{
                var _cho = setInterval(function() {{
                    if (typeof L === 'undefined' || !window['{ten_map}']) return;
                    clearInterval(_cho);
                    var _map = window['{ten_map}'];
                    var _tile = window['{ten_tile}'] || null;

                    var _NutChup = L.Control.extend({{
                        options: {{ position: 'bottomright' }},
                        onAdd: function() {{
                            var btn = L.DomUtil.create('button');
                            btn.innerHTML = '📷';
                            btn.title = 'Tải ảnh bản đồ (độ phân giải cao)';
                            btn.style.cssText = 'width:36px;height:36px;font-size:18px;background:#fff;' +
                                'border:2px solid rgba(0,0,0,0.2);border-radius:4px;cursor:pointer;line-height:1;';
                            L.DomEvent.disableClickPropagation(btn);
                            L.DomEvent.on(btn, 'click', function(e) {{
                                L.DomEvent.stop(e);
                                if (typeof L.easyPrint === 'undefined') {{
                                    alert('Công cụ xuất ảnh chưa tải xong, vui lòng đợi vài giây rồi thử lại.');
                                    return;
                                }}
                                if (!window._easyPrintPlugin_{ten_map}) {{
                                    var _opts = {{
                                        hidden: true,
                                        exportOnly: true,
                                        filename: '{ten_file}',
                                        sizeModes: [{{ name: 'HighRes', width: {rong_px}, height: {cao_px} }}]
                                    }};
                                    if (_tile) {{ _opts.tileLayer = _tile; }}
                                    window._easyPrintPlugin_{ten_map} = L.easyPrint(_opts).addTo(_map);
                                }}
                                window._easyPrintPlugin_{ten_map}.printMap('HighRes', '{ten_file}');
                            }});
                            return btn;
                        }}
                    }});
                    _map.addControl(new _NutChup());
                }}, 300);
            }}

            if (typeof L !== 'undefined' && typeof L.easyPrint !== 'undefined') {{
                _khoiTaoNutChup();
            }} else {{
                _taiScript('https://cdn.jsdelivr.net/npm/leaflet-easyprint@2.1.9/dist/bundle.js', _khoiTaoNutChup, function() {{
                    _taiScript('https://unpkg.com/leaflet-easyprint@2.1.9/dist/bundle.js', _khoiTaoNutChup, _khoiTaoNutChup);
                }});
            }}
        }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(script))


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
        p_str = str(int(r.get('pressure', 0))) if r.get('pressure', 0) > 0 else '-'
        bf_str = f'Cấp {int(bf)}' if bf > 0 else '-'
        rows += f"<tr><td>{t}</td><td>{r.get('lon',0):.1f}E</td><td>{r.get('lat',0):.1f}N</td><td>{bf_str}</td><td>{p_str}</td></tr>"
    return textwrap.dedent(f"""<div class="info-box"><div class="info-title">{title}</div><div class="info-subtitle">{subtitle}</div><table><thead><tr><th>Ngày-Giờ</th><th>Kinh độ</th><th>Vĩ độ</th><th>Cấp gió</th><th>Pmin (hPa)</th></tr></thead><tbody>{rows}</tbody></table></div>""")

def _buoc_luoi_dep(pham_vi):
    """Chọn bước lưới tọa độ (độ) hợp lý để nhãn không bị dày đặc, dựa trên độ rộng vùng hiển thị."""
    for buoc in (1, 2, 5, 10, 15, 20, 30):
        if pham_vi / buoc <= 10:
            return buoc
    return 30

def _dinh_dang_kinh_do(x, pos=None):
    if abs(x) < 1e-9: return "0°"
    return f"{abs(x):.0f}°E" if x > 0 else f"{abs(x):.0f}°W"

def _dinh_dang_vi_do(y, pos=None):
    if abs(y) < 1e-9: return "0°"
    return f"{abs(y):.0f}°N" if y > 0 else f"{abs(y):.0f}°S"

def generate_storm_static_fig(df, title_text, bounds=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#edf3f5')

    if bounds:
        minx, maxx, miny, maxy = bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy']
        bbox_poly = box(minx, miny, maxx, maxy)
    else:
        if not df.empty:
            minx, maxx = df['lon'].min() - 5, df['lon'].max() + 5
            miny, maxy = df['lat'].min() - 5, df['lat'].max() + 5
        else:
            minx, maxx, miny, maxy = 100.0, 120.0, 5.0, 25.0
        bbox_poly = box(minx, miny, maxx, maxy)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    buoc_x = _buoc_luoi_dep(maxx - minx)
    buoc_y = _buoc_luoi_dep(maxy - miny)
    ax.xaxis.set_major_locator(MultipleLocator(buoc_x))
    ax.yaxis.set_major_locator(MultipleLocator(buoc_y))
    ax.xaxis.set_major_formatter(FuncFormatter(_dinh_dang_kinh_do))
    ax.yaxis.set_major_formatter(FuncFormatter(_dinh_dang_vi_do))
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, which='major', linestyle='--', color='#999999', alpha=0.6, zorder=1, linewidth=0.6)

    if os.path.exists(SHP_MASK_PATH):
        try:
            mask = gpd.read_file(SHP_MASK_PATH)
            if mask.crs and mask.crs.to_epsg() != 4326: mask.to_crs(epsg=4326, inplace=True)
            mask = gpd.clip(mask, bbox_poly)
            if not mask.empty:
                mask.plot(ax=ax, color='#f7f7f5', edgecolor='#e0e0e0', linewidth=0.5, zorder=2)
        except: pass
    if os.path.exists(SHP_DISP_PATH):
        try:
            disp = gpd.read_file(SHP_DISP_PATH)
            if disp.crs and disp.crs.to_epsg() != 4326: disp.to_crs(epsg=4326, inplace=True)
            disp = gpd.clip(disp, bbox_poly)
            if not disp.empty:
                disp.plot(ax=ax, facecolor='none', edgecolor='#555555', linewidth=0.8, zorder=3)
        except: pass

    ax.set_xlabel("Kinh độ", fontsize=11)
    ax.set_ylabel("Vĩ độ", fontsize=11)

    if not df.empty:
        groups = df['storm_no'].unique() if 'storm_no' in df.columns else [None]
        for g in groups:
            sub = df[df['storm_no']==g] if g else df
            dense = densify_track(sub)
            f6, f10, fc = create_storm_swaths(dense)

            if f6 and not f6.is_empty:
                gpd.GeoSeries([f6]).plot(ax=ax, color='#ffccd5', alpha=0.5, zorder=4)
            if f10 and not f10.is_empty:
                gpd.GeoSeries([f10]).plot(ax=ax, color='#ffb3a7', alpha=0.6, zorder=4)
            if fc and not fc.is_empty:
                gpd.GeoSeries([fc]).plot(ax=ax, color='#c1f0c1', alpha=0.6, zorder=4)

            ax.plot(sub['lon'], sub['lat'], color='#222222', linewidth=2, zorder=5)

            for _, r in sub.iterrows():
                is_past = 'quá khứ' in str(r.get('status_raw','')).lower() or 'past' in str(r.get('status_raw','')).lower()
                m_color = '#444444' if is_past else '#e60000'
                ax.plot(r['lon'], r['lat'], marker='o', markersize=6, color=m_color, markeredgecolor='white', markeredgewidth=0.8, zorder=6)

    legend_elements = [
        Patch(facecolor='#ffccd5', edgecolor='none', label='Vùng gió mạnh > cấp 6'),
        Patch(facecolor='#ffb3a7', edgecolor='none', label='Vùng gió mạnh > cấp 10'),
        Patch(facecolor='#c1f0c1', edgecolor='none', label='Vùng tâm bão/ATNĐ có thể đi qua'),
        Line2D([0], [0], marker='o', color='w', label='Tâm đã đi qua', markerfacecolor='#444444', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Tâm hiện tại/Dự báo', markerfacecolor='#e60000', markersize=8)
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', title="Chú Thích", title_fontsize=11, fontsize=9, framealpha=0.92, facecolor='white', edgecolor='#dddddd')
    leg.set_zorder(10)

    if not df.empty:
        if 'status_raw' in df.columns:
            cur = df[df['status_raw'].astype(str).str.contains("hiện tại|current", case=False, na=False)]
            fut = df[df['status_raw'].astype(str).str.contains("dự báo|forecast", case=False, na=False)]
            display_df = pd.concat([cur, fut]).head(6)
        else:
            display_df = df.sort_values('dt', ascending=False).groupby('name').head(1)

        if not display_df.empty:
            cell_text = []
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
                cell_text.append([
                    t,
                    f"{r.get('lon',0):.1f}°E",
                    f"{r.get('lat',0):.1f}°N",
                    f"Cấp {int(bf)}" if bf>0 else "-",
                    f"{int(r.get('pressure',0))}" if r.get('pressure',0)>0 else "-"
                ])
            col_labels = ['Ngày-Giờ', 'Kinh độ', 'Vĩ độ', 'Cấp gió', 'Pmin(hPa)']

            table = ax.table(cellText=cell_text, colLabels=col_labels, loc='lower right',
                             bbox=[0.62, 0.03, 0.36, 0.22], cellLoc='center', zorder=10)
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)

            for (row, col), cell in table.get_celld().items():
                cell.set_facecolor('white')
                cell.set_edgecolor('#cccccc')
                if row == 0:
                    cell.set_text_props(weight='bold', color='#333333')
                    cell.set_facecolor('#f8f9fa')

            subtitle = "(Đang cập nhật)"
            target_row = display_df.iloc[0]
            if 'hour_explicit' in target_row and pd.notna(target_row['hour_explicit']):
                subtitle = f"Tin phát lúc {int(target_row['hour_explicit'])}h30"

            title_box_text = f"{title_text}\n{subtitle}"
            ax.text(0.02, 0.98, title_box_text, transform=ax.transAxes, ha="left", va="top",
                    fontsize=12, fontweight='bold', color='#333333',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#dddddd', alpha=0.92, zorder=10))

    return fig

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

def run_interpolation_and_plot(input_df, title_text, data_type='temp'):
    minx, maxx, miny, maxy = 101.8, 115.0, 8.0, 23.9
    GRID_N, SIGMA, IDW_POWER, KNN = 1000, 1.5, 3.0, 12

    if data_type == 'rain':
        vmin, vmax = 0, 1400
        levels_for_ticks = np.arange(0, 1450, 100)
        colors = ['#FFFFFF', '#A0E6FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800080', '#4B0082']
        cmap = LinearSegmentedColormap.from_list('rain_smooth', colors, N=512)
        cmap.set_under(colors[0]); cmap.set_over(colors[-1])
        unit_label = "Lượng mưa (mm)"
    else:
        vmin, vmax = 0.0, 40.0
        levels_for_ticks = list(range(0, 42, 4))
        colors = [(0.0, '#FFFFFF'), (0.1, '#D0F0FF'), (0.2, '#00A0FF'), (0.4, '#00FF00'), (0.6, '#FFFF00'), (0.75, '#FFA500'), (0.9, '#FF0000'), (1.0, '#8B0000')]
        cmap = LinearSegmentedColormap.from_list("custom_smooth_temp", colors, N=256)
        unit_label = "Nhiệt độ (°C)"

    norm = Normalize(vmin=vmin, vmax=vmax)
    input_df.columns = input_df.columns.str.lower().str.strip()
    if not all(c in input_df.columns for c in ['lon', 'lat', 'value']): return None, "File thiếu cột bắt buộc."
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, "Dữ liệu trống."

    x_pts, y_pts, z_pts = valid['lon'].to_numpy(), valid['lat'].to_numpy(), valid['value'].to_numpy()
    edge_points = pd.DataFrame({'lon': [minx, minx, maxx, maxx, (minx + maxx)/2], 'lat': [miny, maxy, miny, maxy, (miny + maxy)/2], 'value': [float(np.nanmean(z_pts))] * 5})
    aug = pd.concat([valid[['lon', 'lat', 'value']], edge_points], ignore_index=True)
    xi, yi, zi = aug['lon'].to_numpy(), aug['lat'].to_numpy(), aug['value'].to_numpy()

    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(xi, yi, zi, grid_xy, k=KNN, power=IDW_POWER).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    mask_shape = disp_shape = None
    if os.path.exists(SHP_MASK_PATH):
        try:
            mask_shape = gpd.read_file(SHP_MASK_PATH)
            if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)
        except Exception as e: return None, f"Lỗi đọc Mask Shapefile: {e}"
    else:
        mask_shape = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

    if os.path.exists(SHP_DISP_PATH):
        try:
            disp_shape = gpd.read_file(SHP_DISP_PATH)
            if disp_shape.crs and disp_shape.crs.to_epsg() != 4326: disp_shape.to_crs(epsg=4326, inplace=True)
        except Exception: pass
    else: disp_shape = mask_shape

    if mask_shape is not None:
        prep_shape = prep(mask_shape.unary_union)
        mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(gx.shape)
        gv_masked = np.where(mask_flat, gv, np.nan)
    else: gv_masked = gv

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(title_text if title_text else f'Bản đồ {unit_label}', fontsize=16)
    if disp_shape is not None: disp_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy], cmap=cmap, norm=norm, interpolation='bilinear', origin='lower')
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, extend='both')
    cbar.set_label(unit_label, fontsize=12)
    cbar.set_ticks(levels_for_ticks)
    cbar.set_ticklabels([str(l) for l in levels_for_ticks])
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.ticklabel_format(useOffset=False, style='plain')
    return fig, None

def generate_single_province_fig(cache, prov_name, title_text):
    mask_shape = cache.get('mask_shape')
    shape_col = cache.get('shape_col', "")

    if mask_shape is None or not shape_col or shape_col not in mask_shape.columns:
        return None

    prov_shape = mask_shape[mask_shape[shape_col] == prov_name]
    if prov_shape.empty: return None

    p_minx, p_miny, p_maxx, p_maxy = prov_shape.total_bounds
    pad_x, pad_y = (p_maxx - p_minx) * 0.1, (p_maxy - p_miny) * 0.1
    p_minx -= pad_x; p_maxx += pad_x; p_miny -= pad_y; p_maxy += pad_y

    grid_xy = np.column_stack([cache['gx'].ravel(), cache['gy'].ravel()])
    prep_shape = prep(prov_shape.unary_union)
    mask_flat = np.fromiter((prep_shape.contains(Point(px, py)) for px, py in grid_xy), count=grid_xy.shape[0], dtype=bool).reshape(cache['gx'].shape)
    gv_masked = np.where(mask_flat, cache['gv'], np.nan)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{title_text}\n(Khu vực: {prov_name})", fontsize=16)
    prov_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)
    im = ax.imshow(gv_masked, extent=[cache['minx'], cache['maxx'], cache['miny'], cache['maxy']], cmap=cache['cmap'], norm=cache['norm'], interpolation='bilinear', origin='lower')
    ax.set_xlim(p_minx, p_maxx); ax.set_ylim(p_miny, p_maxy)

    cbar = plt.colorbar(im, ax=ax, extend='both', shrink=0.7, pad=0.02)
    cbar.set_ticks(cache['custom_levels'])
    cbar.set_ticklabels([f"{val:.1f}" for val in cache['custom_levels']])
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel("Kinh độ"); ax.set_ylabel("Vĩ độ")
    return fig


# ==============================================================================
# HÀM NỘI SUY TƯƠNG TÁC (CÓ LỚP XÃ)
# ==============================================================================
def run_interactive_folium_interpolation(
    input_df, title_text, cmap_name, num_bins, custom_levels,
    selected_provinces, shape_col, custom_bounds=None,
    show_xa_layer=False, show_xa_labels=False
):
    """
    Nội suy IDW → Folium map.
    Tham số mới:
        show_xa_layer  : bool – hiển thị lớp ranh giới xã (RG_xa_VN.shp)
        show_xa_labels : bool – hiển thị nhãn tên xã (cột ten_xa)
    """
    shape_col = shape_col or ""
    input_df.columns = input_df.columns.str.lower().str.strip()
    if not all(c in input_df.columns for c in ['lon', 'lat', 'value']):
        return None, None, None, "File thiếu cột bắt buộc."
    valid = input_df.dropna(subset=['lon', 'lat', 'value']).copy()
    if valid.empty: return None, None, None, "Dữ liệu trống."

    if not os.path.exists(SHP_MASK_PATH): return None, None, None, "Không tìm thấy file vn34tinh.shp"

    mask_shape = gpd.read_file(SHP_MASK_PATH)
    if mask_shape.crs and mask_shape.crs.to_epsg() != 4326: mask_shape.to_crs(epsg=4326, inplace=True)

    if selected_provinces and shape_col and shape_col in mask_shape.columns:
        mask_shape = mask_shape[mask_shape[shape_col].isin(selected_provinces)]
        if mask_shape.empty: return None, None, None, "Không tìm thấy tỉnh đã chọn."

    disp_shape = None
    if os.path.exists(SHP_DISP_PATH):
        try:
            disp_shape = gpd.read_file(SHP_DISP_PATH)
            if disp_shape.crs and disp_shape.crs.to_epsg() != 4326: disp_shape.to_crs(epsg=4326, inplace=True)
        except Exception: pass

    # ---- Lớp ranh giới xã ----
    xa_shape = None
    if show_xa_layer and os.path.exists(SHP_XA_PATH):
        try:
            xa_shape = gpd.read_file(SHP_XA_PATH)
            if xa_shape.crs and xa_shape.crs.to_epsg() != 4326:
                xa_shape = xa_shape.to_crs(epsg=4326)
        except Exception as e:
            st.warning(f"⚠️ Không tải được lớp xã: {e}")
            xa_shape = None

    if custom_bounds:
        minx, maxx, miny, maxy = custom_bounds['minx'], custom_bounds['maxx'], custom_bounds['miny'], custom_bounds['maxy']
        bbox_poly = box(minx, miny, maxx, maxy)
        if not mask_shape.empty: mask_shape = gpd.clip(mask_shape, bbox_poly)
        if disp_shape is not None and not disp_shape.empty: disp_shape = gpd.clip(disp_shape, bbox_poly)
        if xa_shape is not None and not xa_shape.empty: xa_shape = gpd.clip(xa_shape, bbox_poly)
    else:
        minx, miny, maxx, maxy = mask_shape.total_bounds
        minx -= 0.5; maxx += 0.5; miny -= 0.5; maxy += 0.5

    x_pts, y_pts, z_pts = valid['lon'].to_numpy(), valid['lat'].to_numpy(), valid['value'].to_numpy()
    GRID_N, SIGMA = 800, 1.0
    gx, gy = np.meshgrid(np.linspace(minx, maxx, GRID_N), np.linspace(miny, maxy, GRID_N))
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    gv = idw_knn(x_pts, y_pts, z_pts, grid_xy, k=12, power=3.0).reshape(gx.shape)
    if SIGMA > 0: gv = gaussian_filter(gv, sigma=SIGMA)

    shape_union = mask_shape.unary_union
    if shape_union.is_empty:
        mask_flat = np.ones(gx.shape, dtype=bool)
    else:
        prep_shape = prep(shape_union)
        mask_flat = np.fromiter(
            (prep_shape.contains(Point(px, py)) for px, py in grid_xy),
            count=grid_xy.shape[0], dtype=bool
        ).reshape(gx.shape)

    gv_masked = np.where(mask_flat, gv, np.nan)

    cmap = plt.get_cmap(cmap_name)
    if custom_levels is not None and len(custom_levels) > 1:
        custom_levels = sorted(list(set(custom_levels)))
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')
    else:
        vmin_val, vmax_val = np.nanmin(gv_masked), np.nanmax(gv_masked)
        if np.isnan(vmin_val): vmin_val, vmax_val = 0, 1
        custom_levels = np.linspace(vmin_val, vmax_val, num_bins + 1)
        norm = BoundaryNorm(custom_levels, ncolors=cmap.N, extend='both')

    cache_dict = {
        'gv': gv, 'gx': gx, 'gy': gy, 'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy,
        'cmap': cmap, 'norm': norm, 'custom_levels': custom_levels,
        'mask_shape': mask_shape, 'shape_col': shape_col
    }

    rgba = cmap(norm(gv_masked))
    rgba[np.isnan(gv_masked)] = [0, 0, 0, 0]
    rgba_folium = np.flipud(rgba)

    buf = io.BytesIO()
    plt.imsave(buf, rgba_folium, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()

    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], tiles=None)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # ---- Các lớp nền (có thể bật/tắt qua LayerControl) ----
    folium.TileLayer('CartoDB positron',   name='🗺️ Nền Sáng (CartoDB)',    overlay=False, control=True, show=True).add_to(m)
    folium.TileLayer('OpenStreetMap',      name='🗺️ OpenStreetMap',          overlay=False, control=True, show=False).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='🛰️ Vệ tinh (Esri)',
        overlay=False, control=True, show=False
    ).add_to(m)
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap', name='🏔️ Địa hình (Topo)',
        overlay=False, control=True, show=False
    ).add_to(m)

    # ---- Lớp nội suy ----
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_base64}",
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=0.75, name='🎨 Lớp nội suy', interactive=False
    ).add_to(m)

    # ---- Ranh giới vùng/quốc gia ----
    if disp_shape is not None and not disp_shape.empty:
        folium.GeoJson(
            disp_shape, name="🌏 Ranh giới Khu vực/Quốc gia",
            style_function=lambda x: {'fillColor': 'transparent', 'color': '#333333', 'weight': 1.5, 'dashArray': '4, 4'},
            interactive=False
        ).add_to(m)

    # ---- Ranh giới tỉnh ----
    tooltip_fields = [shape_col] if shape_col and shape_col in mask_shape.columns else []
    tooltip_aliases = ['Tên Tỉnh: '] if tooltip_fields else []
    if not mask_shape.empty:
        folium.GeoJson(
            mask_shape, name="🏛️ Ranh giới Tỉnh",
            style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1.0},
            highlight_function=lambda x: {'weight': 3, 'color': 'red', 'fillColor': '#ff0000', 'fillOpacity': 0.2},
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases) if tooltip_fields else None
        ).add_to(m)

    # ================================================================
    # LỚP RANH GIỚI XÃ (bật/tắt độc lập)
    # ================================================================
    if show_xa_layer and xa_shape is not None and not xa_shape.empty:
        xa_col = 'ten_xa' if 'ten_xa' in xa_shape.columns else None

        # Tooltip hiển thị tên xã khi hover
        xa_tooltip = None
        if xa_col:
            xa_tooltip = folium.GeoJsonTooltip(
                fields=[xa_col],
                aliases=['Tên xã: '],
                sticky=True,
                style="font-size:11px; background-color:white; border:1px solid #ccc; padding:4px;"
            )

        fg_xa = folium.FeatureGroup(name="🏘️ Ranh giới Xã (RG_xa_VN.shp)", show=True)
        folium.GeoJson(
            xa_shape,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': '#8B4513',   # nâu đất – phân biệt rõ với lớp tỉnh đen
                'weight': 0.6,
                'dashArray': '2, 3'
            },
            highlight_function=lambda x: {
                'weight': 2, 'color': '#FF8C00', 'fillColor': '#FFA500', 'fillOpacity': 0.15
            },
            tooltip=xa_tooltip
        ).add_to(fg_xa)

        # ---- Nhãn tên xã (DivIcon) – chỉ hiển thị khi bật ----
        if show_xa_labels and xa_col:
            fg_xa_labels = folium.FeatureGroup(name="🏷️ Tên Xã", show=True)
            for _, row_xa in xa_shape.iterrows():
                try:
                    geom = row_xa.geometry
                    if geom is None or geom.is_empty: continue
                    # Lấy centroid để đặt nhãn
                    cx, cy = geom.centroid.x, geom.centroid.y
                    ten = str(row_xa.get(xa_col, ''))
                    if not ten or ten == 'nan': continue
                    folium.Marker(
                        location=[cy, cx],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size:8px; font-weight:600; color:#5C2C00; '
                                 f'white-space:nowrap; text-shadow: 0 0 3px white, 0 0 3px white;">'
                                 f'{ten}</div>',
                            icon_size=(80, 15),
                            icon_anchor=(40, 7)
                        )
                    ).add_to(fg_xa_labels)
                except Exception:
                    continue
            fg_xa_labels.add_to(m)

        fg_xa.add_to(m)

    # ---- Thanh màu chú thích ----
    colormap_branca = cm.StepColormap(
        colors=[mcolors.to_hex(cmap(norm(val))) for val in custom_levels[:-1]],
        vmin=custom_levels[0], vmax=custom_levels[-1],
        index=custom_levels, caption=title_text
    )
    m.add_child(colormap_branca)
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)

    # ---- Ảnh tĩnh matplotlib (để tải về) ----
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title_text, fontsize=16)
    if disp_shape is not None and not disp_shape.empty:
        disp_shape.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0)
    if not mask_shape.empty:
        mask_shape.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.5, linestyle=':')
    if show_xa_layer and xa_shape is not None and not xa_shape.empty:
        xa_shape.boundary.plot(ax=ax, edgecolor='#8B4513', linewidth=0.4, linestyle='--', alpha=0.7)
        if show_xa_labels and xa_col:
            for _, row_xa in xa_shape.iterrows():
                try:
                    geom = row_xa.geometry
                    if geom is None or geom.is_empty: continue
                    cx, cy = geom.centroid.x, geom.centroid.y
                    ten = str(row_xa.get(xa_col, ''))
                    if not ten or ten == 'nan': continue
                    ax.text(cx, cy, ten, fontsize=5, ha='center', va='center',
                            color='#5C2C00', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.6))
                except Exception:
                    continue
    im = ax.imshow(gv_masked, extent=[minx, maxx, miny, maxy],
                   cmap=cmap, norm=norm, interpolation='bilinear', origin='lower')
    cbar = plt.colorbar(im, ax=ax, extend='both', shrink=0.7, pad=0.02)
    cbar.set_ticks(custom_levels)
    cbar.set_ticklabels([f"{val:.1f}" for val in custom_levels])
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel("Kinh độ"); ax.set_ylabel("Vĩ độ")

    return m, fig, cache_dict, None


# ==============================================================================
# 3b. BỘ GIẢI MÃ ĐIỆN KHÍ TƯỢNG (SYNOP / METAR) - TÍNH NĂNG MỚI
# Dựa theo QCVN 16:2008/BTNMT - Quy chuẩn kỹ thuật quốc gia về Mã luật khí tượng bề mặt
# ==============================================================================
# -*- coding: utf-8 -*-
import re

# ------------------------------------------------------------------
# CÁC BẢNG MÃ (trích từ Phụ lục 2 - QCVN 16:2008/BTNMT)
# ------------------------------------------------------------------

# Bảng mã 2700 - N/Nh: lượng mây (bát phần / oktas)
BANG_N = {
    '0': "Không có mây (trời quang)", '1': "1/10 hoặc ít hơn (gần như quang)",
    '2': "2/10 - 3/10", '3': "4/10", '4': "5/10", '5': "6/10",
    '6': "7/10 - 8/10", '7': "9/10 hoặc nhiều hơn nhưng chưa kín trời",
    '8': "10/10 (kín trời)", '9': "Trời tối do sương mù/hiện tượng khác (không xác định lượng mây)",
    '/': "Không quan trắc được lượng mây",
}

# Bảng mã 2700 - dạng rút gọn (phân số) dùng cho bảng xuất Excel
BANG_N_NGAN = {
    '0': "0/10", '1': "1/10", '2': "2-3/10", '3': "4/10", '4': "5/10",
    '5': "6/10", '6': "7-8/10", '7': "9/10", '8': "10/10",
    '9': "Che khuất", '/': "Không quan trắc",
}

# Bảng mã 1600 - h: độ cao chân mây thấp nhất (đơn giản hoá)
BANG_h = {
    '0': "0 - 50 m", '1': "50 - 100 m", '2': "100 - 200 m", '3': "200 - 300 m",
    '4': "300 - 600 m", '5': "600 - 1000 m", '6': "1000 - 1500 m",
    '7': "1500 - 2000 m", '8': "2000 - 2500 m", '9': "≥ 2500 m hoặc không có mây",
    '/': "Không xác định được (chân mây thấp hơn mực trạm/bị che khuất)",
}

# Bảng mã 0200 - a: đặc điểm khuynh hướng khí áp 3 giờ qua
BANG_a = {
    '0': "Tăng rồi giảm (khí áp hiện tại bằng hoặc cao hơn 3 giờ trước)",
    '1': "Tăng rồi giữ nguyên, hoặc tăng chậm dần (khí áp hiện tại cao hơn 3 giờ trước)",
    '2': "Tăng đều hoặc không đều",
    '3': "Giảm hoặc giữ nguyên rồi tăng; hoặc tăng nhanh dần",
    '4': "Giữ nguyên (khí áp không đổi so với 3 giờ trước)",
    '5': "Giảm rồi tăng (khí áp hiện tại bằng hoặc thấp hơn 3 giờ trước)",
    '6': "Giảm rồi giữ nguyên, hoặc giảm chậm dần (khí áp hiện tại thấp hơn 3 giờ trước)",
    '7': "Giảm đều hoặc không đều",
    '8': "Giữ nguyên hoặc tăng rồi giảm; hoặc giảm nhanh dần",
}

# Bảng mã 0200 - dạng rút gọn (không kèm chú giải trong ngoặc) dùng cho bảng xuất Excel
BANG_a_NGAN = {
    '0': "Tăng rồi giảm", '1': "Tăng rồi giữ nguyên, hoặc tăng chậm dần",
    '2': "Tăng đều hoặc không đều", '3': "Giảm hoặc giữ nguyên rồi tăng; hoặc tăng nhanh dần",
    '4': "Giữ nguyên", '5': "Giảm rồi tăng",
    '6': "Giảm rồi giữ nguyên, hoặc giảm chậm dần", '7': "Giảm đều hoặc không đều",
    '8': "Giữ nguyên hoặc tăng rồi giảm; hoặc giảm nhanh dần",
}

# Bảng mã 4019 - tR: thời đoạn tính lượng giáng thủy
BANG_tR = {
    '0': "không xác định/không kết thúc đúng kỳ quan trắc", '1': "6 giờ", '2': "12 giờ",
    '3': "18 giờ", '4': "24 giờ", '5': "1 giờ", '6': "2 giờ", '7': "3 giờ",
    '8': "9 giờ", '9': "15 giờ",
}

# Bảng mã 0500 - C: loại mây (chung)
BANG_C = {
    '0': "Ti (Cirrus, Ci)", '1': "Ti tích (Cirrocumulus, Cc)", '2': "Ti tầng (Cirrostratus, Cs)",
    '3': "Trung tích (Altocumulus, Ac)", '4': "Trung tầng (Altostratus, As)",
    '5': "Vũ tầng (Nimbostratus, Ns)", '6': "Tầng tích (Stratocumulus, Sc)",
    '7': "Tầng (Stratus, St)", '8': "Tích (Cumulus, Cu)", '9': "Vũ tích (Cumulonimbus, Cb)",
    '/': "Không nhìn thấy mây (trời tối/sương mù/bão cát...)",
}

# Bảng mã 0513 - CL (rút gọn thuyết minh thông thường)
BANG_CL = {
    '0': "Không có mây CL (Sc/St/Cu/Cb)",
    '1': "Cu dạng dẹt (humilis/fractus), không phải trời xấu",
    '2': "Cu phát triển vừa/mạnh (mediocris/congestus), có thể kèm Sc",
    '3': "Cb dạng calvus (đỉnh mờ, chưa có dạng đe rõ)",
    '4': "Sc hình thành từ Cu tỏa ra (cumulogenitus)",
    '5': "Sc không phải do Cu tỏa ra",
    '6': "St dạng màn/lớp, không phải trời xấu",
    '7': "Mảnh St hoặc Cu trời xấu (dưới As/Ns)",
    '8': "Cu và Sc không cùng mực chân mây (không do Cu tỏa ra)",
    '9': "Cb dạng capillatus (có đe rõ, dạng sợi ở đỉnh)",
    '/': "Không thấy được mây CL (trời tối/sương mù/che khuất)",
}

# Bảng mã 0515 - CM (rút gọn)
BANG_CM = {
    '0': "Không có mây CM (Ac/As/Ns)",
    '1': "As bán trong suốt (translucidus)",
    '2': "As dày đặc (opacus) hoặc Ns",
    '3': "Ac một mực, bán trong suốt",
    '4': "Ac dạng đám (thấu kính/hình cá), biến đổi hình dạng",
    '5': "Ac thành dải/nhiều lớp, xâm chiếm dần bầu trời",
    '6': "Ac hình thành từ Cu/Cb tỏa ra (cumulogenitus)",
    '7': "Ac ở 2 lớp trở lên, hoặc cùng As/Ns",
    '8': "Ac dạng castellanus/floccus (sùi hình tháp nhỏ)",
    '9': "Ac trong bầu trời hỗn độn, nhiều mực cao",
    '/': "Không thấy được mây CM (trời tối/sương mù/bị che khuất)",
}

# Bảng mã 0509 - CH (rút gọn)
BANG_CH = {
    '0': "Không có mây CH (Ci/Cs)",
    '1': "Ci dạng tơ sợi/móc câu, chưa xâm chiếm bầu trời",
    '2': "Ci dày thành đám/bó (có thể là tàn dư đỉnh Cb)",
    '3': "Ci dày đặc dạng đe (tàn dư đỉnh Cb)",
    '4': "Ci móc câu/tơ sợi, đang xâm chiếm dần bầu trời",
    '5': "Ci và/hoặc Cs xâm chiếm dần, chưa quá 45° chân trời",
    '6': "Ci và/hoặc Cs xâm chiếm dần, đã quá 45° chân trời",
    '7': "Cs phủ kín toàn bộ bầu trời",
    '8': "Cs không xâm chiếm dần, không phủ kín trời",
    '9': "Chủ yếu là Cc",
    '/': "Không thấy được mây CH (trời tối/sương mù/bị che khuất)",
}

# Bảng mã 0901 - E: trạng thái mặt đất (không có tuyết/băng)
BANG_E = {
    '0': "Mặt đất khô", '1': "Mặt đất ẩm", '2': "Mặt đất ướt (có vũng nước)",
    '3': "Ngập nước", '4': "Mặt đất đông giá", '5': "Mặt đất có váng băng",
    '6': "Bụi/cát tơi khô, chưa phủ kín mặt đất", '7': "Lớp bụi/cát tơi mỏng phủ kín mặt đất",
    '8': "Lớp bụi/cát tơi trung bình hoặc dày phủ kín mặt đất", '9': "Đất cực khô, có khe nứt",
}

# Bảng mã 0877 - dd: hướng gió (đơn vị chục độ)
def mota_dd(dd_str):
    if dd_str == "00": return "Lặng gió (không có hướng)"
    if dd_str == "99": return "Hướng gió biến đổi"
    try:
        deg = int(dd_str) * 10
        return f"~{deg}°"
    except Exception:
        return "Không xác định"

# Bảng mã 4377 - VV: tầm nhìn ngang (mã số -> km)
def mota_VV(vv_str):
    try:
        v = int(vv_str)
    except Exception:
        return "Không xác định"
    if 0 <= v <= 50:
        return f"{v/10:.1f} km"
    if v == 56: return "6 km"
    if 57 <= v <= 80:
        table_57_80 = {57:7,58:8,59:9,60:10,61:11,62:12,63:13,64:14,65:15,66:16,67:17,68:18,69:19,
                       70:20,71:21,72:22,73:23,74:24,75:25,76:26,77:27,78:28,79:29,80:30}
        return f"{table_57_80.get(v,'?')} km"
    table_81_89 = {81:35,82:40,83:45,84:50,85:55,86:60,87:65,88:70}
    if v in table_81_89: return f"{table_81_89[v]} km"
    if v == 89: return "> 70 km"
    table_90_99 = {90:"< 0,05 km",91:"0,05 km",92:"0,2 km",93:"0,5 km",94:"1 km",
                   95:"2 km",96:"4 km",97:"10 km",98:"20 km",99:"≥ 50 km"}
    if v in table_90_99: return table_90_99[v]
    return "Không dùng"

# Bảng mã 4377 - VV: tầm nhìn ngang, dạng số (km) dùng cho bảng xuất Excel
def mota_VV_km(vv_str):
    try:
        v = int(vv_str)
    except Exception:
        return None
    if 0 <= v <= 50: return round(v / 10.0, 1)
    if v == 56: return 6.0
    if 57 <= v <= 80:
        table_57_80 = {57:7,58:8,59:9,60:10,61:11,62:12,63:13,64:14,65:15,66:16,67:17,68:18,69:19,
                       70:20,71:21,72:22,73:23,74:24,75:25,76:26,77:27,78:28,79:29,80:30}
        return float(table_57_80.get(v)) if v in table_57_80 else None
    table_81_89 = {81:35,82:40,83:45,84:50,85:55,86:60,87:65,88:70,89:70}
    if v in table_81_89: return float(table_81_89[v])
    table_90_99 = {90:0.05,91:0.05,92:0.2,93:0.5,94:1.0,95:2.0,96:4.0,97:10.0,98:20.0,99:50.0}
    if v in table_90_99: return table_90_99[v]
    return None


def mota_hshs(code_str):
    try:
        v = int(code_str)
    except Exception:
        return "Không xác định"
    if 0 <= v <= 49:
        if v == 0: return "< 30 m"
        return f"~{v*30} m"
    table_56_89 = {56:1800,57:2100,58:2400,59:2700,60:3000,61:3300,62:3600,63:3900,64:4200,
                   65:4500,66:4800,67:5100,68:5400,69:5700,70:6000,71:6300,72:6600,73:6900,
                   74:7200,75:7500,76:7800,77:8100,78:8400,79:8700,80:9000,81:10500,82:12000,
                   83:13500,84:15000,85:16500,86:18000,87:19500,88:21000}
    if v in table_56_89: return f"~{table_56_89[v]} m"
    if v == 89: return "> 21000 m"
    table_90_99 = {90:"< 50 m",91:"50 - 100 m",92:"100 - 200 m",93:"200 - 300 m",94:"300 - 600 m",
                   95:"600 - 1000 m",96:"1000 - 1500 m",97:"1500 - 2000 m",98:"2000 - 2500 m",
                   99:"≥ 2500 m hoặc không có mây"}
    if v in table_90_99: return table_90_99[v]
    return "Không dùng"

# Bảng mã 3590 - RRR: lượng giáng thủy (mm)
def mota_RRR(rrr_str):
    if rrr_str == "///": return None, "Không đo được lượng mưa"
    try:
        v = int(rrr_str)
    except Exception:
        return None, "Không xác định"
    if v == 0: return 0.0, "Không có giáng thủy"
    if 1 <= v <= 988: return float(v), f"{v} mm"
    if v == 989: return None, "≥ 989 mm"
    if v == 990: return 0.0, "Lượng giáng thủy dạng giọt (< 0,1mm)"
    if 991 <= v <= 999: return round((v-990)/10.0, 1), f"{(v-990)/10:.1f} mm"
    return None, "Không xác định"

# Danh sách trạm khí tượng bề mặt Việt Nam (Phụ lục 3, QCVN 16:2008/BTNMT)
# Số hiệu trạm iii (3 số cuối của biểu số WMO 48iii). Danh sách rút gọn,
# tra cứu tốt nhất - có thể chưa đầy đủ 100% do định dạng bảng gốc.
VN_STATIONS = {
    "800": "Lai Châu", "802": "Sa Pa", "803": "Lào Cai", "805": "Hà Giang",
    "806": "Sơn La", "807": "Thất Khê", "808": "Cao Bằng", "809": "Bắc Giang",
    "810": "Bắc Cạn", "811": "Điện Biên", "812": "Tuyên Quang", "813": "Việt Trì",
    "814": "Vĩnh Yên", "815": "Yên Bái", "816": "Hoài Đức", "817": "Sơn Tây",
    "818": "Hòa Bình", "820": "Láng (Hà Nội)", "821": "Hà Nam (Phủ Lý)",
    "822": "Hưng Yên", "823": "Nam Định", "824": "Ninh Bình", "826": "Phù Liễn",
    "827": "Hải Dương", "828": "Hòn Dấu", "829": "Văn Lý", "830": "Lạng Sơn",
    "831": "Thái Nguyên", "832": "Nho Quan", "833": "Bãi Cháy", "834": "Cô Tô",
    "835": "Thái Bình", "836": "Cửa Ông", "837": "Tiên Yên", "838": "Móng Cái",
    "839": "Bạch Long Vĩ", "840": "Thanh Hóa", "842": "Hồi Xuân", "845": "Vinh",
    "846": "Hà Tĩnh", "847": "Ba Đồn", "848": "Đồng Hới", "849": "Đông Hà",
    "852": "Huế", "855": "Đà Nẵng", "860": "Hoàng Sa", "861": "Đắc Tô",
    "863": "Quảng Ngãi", "864": "An Nhơn", "865": "Kon Tum", "866": "Pleiku",
    "867": "An Khê", "868": "Ialy", "869": "Eakmat", "870": "Quy Nhơn",
    "872": "Ayunpa (Cheo Reo)", "873": "Tuy Hòa", "875": "Buôn Ma Thuột",
    "876": "Eahleo", "877": "Nha Trang", "878": "Buôn Hồ", "879": "Cam Ranh",
    "880": "Đà Lạt", "881": "Liên Khương", "882": "Đăk Mil", "883": "Phước Long",
    "884": "Bảo Lộc", "885": "Lăk", "886": "Đak Nông", "887": "Phan Thiết",
    "888": "La Gi (Hàm Tân)", "889": "Phú Quý (Cù Lao Thu)", "890": "Phan Rang",
    "892": "Song Tử Tây", "895": "Đồng Phú", "896": "Biên Hòa", "898": "Tây Ninh",
    "899": "Sở Sao (Thủ Dầu Một)", "900": "Tân Sơn Nhất", "901": "Bến Tre",
    "902": "Ba Tri", "903": "Vũng Tàu", "904": "Càng Long", "905": "Vị Thanh",
    "906": "Mộc Hóa", "907": "Rạch Giá", "908": "Cao Lãnh", "909": "Châu Đốc",
    "910": "Cần Thơ", "911": "Vĩnh Long", "912": "Mỹ Tho", "913": "Sóc Trăng",
    "914": "Cà Mau", "915": "Bạc Liêu", "916": "Thổ Chu", "917": "Phú Quốc",
    "918": "Côn Đảo", "919": "Huyền Trân (DK1.7)", "920": "Trường Sa",
}

def ten_tram_synop(iiiii):
    """iiiii: biểu số WMO 5 chữ số, ví dụ '48823'."""
    iiiii = iiiii.strip()
    if len(iiiii) == 5 and iiiii.startswith("48"):
        ma = iiiii[2:]
        ten = VN_STATIONS.get(ma)
        if ten: return ten
    return None

# Sân bay Việt Nam thường gặp trong bản tin METAR (mã ICAO)
VN_AIRPORTS = {
    "VVNB": "Nội Bài (Hà Nội)", "VVTS": "Tân Sơn Nhất (TP.HCM)",
    "VVDN": "Đà Nẵng", "VVCR": "Cam Ranh (Khánh Hòa)",
    "VVDB": "Điện Biên Phủ", "VVCI": "Cát Bi (Hải Phòng)",
    "VVPQ": "Phú Quốc", "VVCT": "Cần Thơ", "VVCA": "Cà Mau",
    "VVVH": "Vinh (Nghệ An)", "VVDL": "Liên Khương (Đà Lạt)",
    "VVPB": "Pleiku", "VVBM": "Buôn Ma Thuột", "VVPC": "Phù Cát (Quy Nhơn)",
    "VVTH": "Thọ Xuân (Thanh Hóa)", "VVRG": "Rạch Giá", "VVCS": "Côn Đảo",
    "VVNT": "Nà Sản (Sơn La)", "VVVD": "Vân Đồn (Quảng Ninh)",
}


def _sn_val(sn, magnitude):
    """Áp dụng dấu (Bảng mã 3845): 0=dương/bằng 0, 1=âm."""
    if sn == '1':
        return -magnitude
    return magnitude


def decode_synop(raw):
    """
    Giải mã bản tin SYNOP dạng FM12 (AAXX) theo QCVN 16:2008/BTNMT.
    Trả về (tom_tat: dict, chi_tiet: list[(nhom_goc, dien_giai)], ghi_chu: list[str]).
    """
    text = raw.strip().replace("=", " ")
    tokens = [t for t in re.split(r"\s+", text) if t]
    if not tokens:
        return {}, [], ["Bản tin trống."]

    chi_tiet = []
    ghi_chu = []
    tom_tat = {}

    idx = 0
    loai = tokens[idx]; idx += 1
    if loai not in ("AAXX", "BBXX", "OOXX"):
        ghi_chu.append(f"Không nhận diện được chỉ báo loại bản tin '{loai}' (cần AAXX/BBXX/OOXX).")
    chi_tiet.append((loai, {
        "AAXX": "Bản tin SYNOP từ trạm cố định trên mặt đất (FM12)",
        "BBXX": "Bản tin SHIP từ trạm trên biển (FM13)",
        "OOXX": "Bản tin SYNOP MOBIL từ trạm di động trên mặt đất (FM14)",
    }.get(loai, "Không xác định loại bản tin")))
    tom_tat["loai_ban_tin"] = loai

    if loai != "AAXX":
        ghi_chu.append("Công cụ hiện hỗ trợ chi tiết nhất cho bản tin AAXX (trạm cố định trên đất); "
                        "với BBXX/OOXX một số nhóm đầu (vị trí tàu/trạm di động) có thể chưa được giải mã chi tiết.")

    # Nhóm YYGGiw
    if idx < len(tokens) and len(tokens[idx]) == 5:
        g = tokens[idx]; idx += 1
        yy, gg, iw = g[0:2], g[2:4], g[4]
        iw_desc = {"0": "ước lượng, đơn vị m/s", "1": "đo bằng máy, đơn vị m/s",
                   "2": "ước lượng, đơn vị knot", "3": "đo bằng máy, đơn vị knot"}.get(iw, "không xác định")
        chi_tiet.append((g, f"Ngày {yy}, giờ quan trắc {gg}:00 UTC (giờ quốc tế - GQT); "
                             f"tốc độ gió {iw_desc}"))
        tom_tat["ngay"] = yy
        tom_tat["gio_utc"] = gg
        tom_tat["don_vi_gio_iw"] = iw
        try:
            tom_tat["ngay_so"] = int(yy)
            tom_tat["gio_so"] = int(gg)
        except Exception:
            pass

    # Nhóm IIiii - số hiệu trạm
    if idx < len(tokens) and tokens[idx].isdigit() and len(tokens[idx]) == 5:
        g = tokens[idx]; idx += 1
        ten_tram = ten_tram_synop(g)
        if ten_tram:
            chi_tiet.append((g, f"Số hiệu trạm WMO: {g} - Trạm {ten_tram}"))
            tom_tat["tram"] = f"{ten_tram} ({g})"
            tom_tat["tram_ten"] = ten_tram
        else:
            chi_tiet.append((g, f"Số hiệu trạm WMO: {g}"))
            tom_tat["tram"] = g
            tom_tat["tram_ten"] = g

    # Nhóm iRixhVV
    if idx < len(tokens) and len(tokens[idx]) == 5:
        g = tokens[idx]; idx += 1
        iR, ix, h, vv = g[0], g[1], g[2], g[3:5]
        chi_tiet.append((g,
            f"iR={iR} (cách phát báo giáng thủy - Bảng mã 1819); "
            f"ix={ix} (kiểu trạm/thời tiết hiện tại - Bảng mã 1860); "
            f"Độ cao chân mây thấp nhất h={h}: {BANG_h.get(h, 'không xác định')}; "
            f"Tầm nhìn ngang VV={vv}: {mota_VV(vv)}"))
        tom_tat["tam_nhin"] = mota_VV(vv)
        tom_tat["tam_nhin_so"] = mota_VV_km(vv)
        tom_tat["do_cao_chan_may_thap_nhat"] = BANG_h.get(h, "?")

    # Nhóm Nddff
    if idx < len(tokens) and len(tokens[idx]) == 5:
        g = tokens[idx]; idx += 1
        n, dd, ff = g[0], g[1:3], g[3:5]
        don_vi = "knot" if tom_tat.get("don_vi_gio_iw") in ("2", "3") else "m/s"
        chi_tiet.append((g,
            f"Tổng lượng mây N={n}: {BANG_N.get(n, 'không xác định')}; "
            f"Hướng gió dd={dd}: {mota_dd(dd)}; Tốc độ gió ff={ff} {don_vi}"))
        tom_tat["may_tong_quan"] = BANG_N.get(n, "?")
        tom_tat["may_tong_quan_ngan"] = BANG_N_NGAN.get(n, "")
        tom_tat["huong_gio"] = mota_dd(dd)
        tom_tat["toc_do_gio"] = f"{int(ff)} {don_vi}" if ff.isdigit() else ff
        if dd.isdigit() and dd not in ("00", "99"):
            tom_tat["huong_gio_so"] = int(dd)
        if ff.isdigit():
            tom_tat["toc_do_gio_so"] = int(ff)

    # Các nhóm còn lại của Đoạn 1, Đoạn 2 (222), Đoạn 3 (333), Đoạn 4 (444), Đoạn 5 (555)
    doan_hien_tai = 1
    while idx < len(tokens):
        g = tokens[idx]; idx += 1

        if g in ("222", "333", "444", "555", "999"):
            doan_hien_tai = int(g[0])
            chi_tiet.append((g, f"--- Bắt đầu Đoạn {doan_hien_tai} ---"))
            continue

        if g == "80000":
            chi_tiet.append((g, "Nhóm báo hiệu bắt đầu các nhóm bổ sung theo quy định khu vực/quốc gia "
                                 "(nội dung tiếp theo cần tra cứu quy ước riêng, chưa giải mã chi tiết)."))
            continue

        try:
            if doan_hien_tai == 1:
                dg = _giai_ma_doan1(g, tom_tat)
            elif doan_hien_tai == 3:
                dg = _giai_ma_doan3(g, tom_tat)
            else:
                dg = None
        except Exception:
            dg = None

        if dg:
            chi_tiet.append((g, dg))
        else:
            chi_tiet.append((g, "Nhóm chưa được hỗ trợ giải mã chi tiết trong công cụ này (cần tra Phụ lục 2 - QCVN 16:2008/BTNMT)."))

    return tom_tat, chi_tiet, ghi_chu


def _giai_ma_doan1(g, tom_tat):
    d0 = g[0]
    if d0 == '1' and len(g) == 5:
        sn, ttt = g[1], g[2:5]
        val = _sn_val(sn, int(ttt) / 10.0)
        tom_tat["nhiet_do_khong_khi"] = f"{val:.1f}°C"
        tom_tat["nhiet_do_so"] = val
        return f"Nhiệt độ không khí = {val:.1f}°C"
    if d0 == '2' and len(g) == 5:
        sn, tdtdtd = g[1], g[2:5]
        if sn == '9':
            return f"Độ ẩm tương đối = {int(tdtdtd)}%"
        val = _sn_val(sn, int(tdtdtd) / 10.0)
        tom_tat["diem_suong"] = f"{val:.1f}°C"
        tom_tat["diem_suong_so"] = val
        return f"Nhiệt độ điểm sương = {val:.1f}°C"
    if d0 == '3' and len(g) == 5:
        pppp = g[1:5]
        p = int(pppp) / 10.0
        if p < 500: p += 1000
        tom_tat["khi_ap_tram"] = f"{p:.1f} hPa"
        tom_tat["khi_ap_tram_so"] = p
        return f"Khí áp mực trạm = {p:.1f} hPa"
    if d0 == '4' and len(g) == 5:
        pppp = g[1:5]
        p = int(pppp) / 10.0
        if p < 500: p += 1000
        tom_tat["khi_ap_mbien"] = f"{p:.1f} hPa"
        tom_tat["khi_ap_mbien_so"] = p
        return f"Khí áp quy về mực nước biển (QNH tương đương) = {p:.1f} hPa"
    if d0 == '5' and len(g) == 5:
        a, ppp = g[1], g[2:5]
        thaydoi = int(ppp) / 10.0
        tom_tat["xu_the_khi_ap"] = f"{BANG_a.get(a,'?')}, biến thiên {thaydoi:.1f} hPa/3h"
        tom_tat["xu_the_khi_ap_ngan"] = BANG_a_NGAN.get(a, "")
        return f"Xu thế khí áp 3 giờ qua: {BANG_a.get(a, 'không xác định')}; biến thiên = {thaydoi:.1f} hPa"
    if d0 == '6' and len(g) == 5:
        rrr, tr = g[1:4], g[4]
        val, mota = mota_RRR(rrr)
        khoang = BANG_tR.get(tr, "?")
        tom_tat["giang_thuy"] = f"{mota} (trong {khoang})"
        if val is not None:
            tom_tat["mua_so"] = val
        return f"Lượng giáng thủy trong {khoang} qua: {mota}"
    if d0 == '8' and len(g) == 5:
        nh, cl, cm, ch = g[1], g[2], g[3], g[4]
        tom_tat["may_ha_trung_cao"] = (f"CL: {BANG_CL.get(cl,'?')}; CM: {BANG_CM.get(cm,'?')}; "
                                        f"CH: {BANG_CH.get(ch,'?')}")
        return (f"Lượng mây Nh={nh}: {BANG_N.get(nh, '?')} | "
                f"Mây tầng thấp CL={cl}: {BANG_CL.get(cl, '?')} | "
                f"Mây tầng trung CM={cm}: {BANG_CM.get(cm, '?')} | "
                f"Mây tầng cao CH={ch}: {BANG_CH.get(ch, '?')}")
    if d0 == '9' and len(g) == 5:
        gg, gg2 = g[1:3], g[3:5]
        return f"Giờ quan trắc chính xác: {gg}:{gg2} UTC"
    return None


def _giai_ma_doan3(g, tom_tat):
    d0 = g[0]
    if d0 == '0' and len(g) == 5:
        e, sn, tgtg = g[1], g[2], g[3:5]
        if e == '/':
            return "Nhóm 0/ThờiTiết: mặt đất có tuyết/băng phủ (không báo trạng thái mặt đất trần)."
        val = _sn_val(sn, int(tgtg))
        return (f"Trạng thái mặt đất lúc quan trắc: {BANG_E.get(e,'?')}; "
                f"nhiệt độ mặt đất lúc quan trắc = {val}°C")
    if d0 == '1' and len(g) == 5:
        sn, txtxtx = g[1], g[2:5]
        val = _sn_val(sn, int(txtxtx) / 10.0)
        return f"Nhiệt độ không khí tối cao (12 giờ qua, ban ngày) = {val:.1f}°C"
    if d0 == '2' and len(g) == 5:
        sn, tntntn = g[1], g[2:5]
        val = _sn_val(sn, int(tntntn) / 10.0)
        tom_tat["nhiet_do_toi_thap"] = f"{val:.1f}°C"
        return f"Nhiệt độ không khí tối thấp (12 giờ qua, ban đêm) = {val:.1f}°C"
    if d0 == '3' and len(g) == 5:
        # Quy ước Việt Nam: nhóm 3Ejjj có dạng 3/SnTgTg
        if g[1] == '/':
            sn, tgtg = g[2], g[3:5]
            val = _sn_val(sn, int(tgtg))
            tom_tat["nhiet_do_toi_thap_mat_dat"] = f"{val}°C"
            return f"[Quy ước VN] Nhiệt độ mặt đất tối thấp đêm qua = {val}°C"
        return "Nhóm 3Ejjj (trạng thái mặt đất có tuyết/băng, khu vực khác VN) - chưa hỗ trợ giải mã chi tiết."
    if d0 == '5' and len(g) == 5:
        two = g[0:2]
        if two == '59':
            p24 = int(g[2:5]) / 10.0
            return f"Biến áp mặt đất 24 giờ qua = -{p24:.1f} hPa (giảm so với 24 giờ trước)"
        if two == '58':
            p24 = int(g[2:5]) / 10.0
            return f"Biến áp mặt đất 24 giờ qua = +{p24:.1f} hPa (tăng so với 24 giờ trước)"
        if g[1] == '5':
            sss = g[2:5]
            return f"Tổng số giờ nắng trong ngày = {int(sss)/10:.1f} giờ"
        return "Nhóm bổ sung 5j1j2j3j4 (Bảng mã 2061) - chưa hỗ trợ giải mã chi tiết cho biến thể này."
    if d0 == '6' and len(g) == 5:
        rrr, tr = g[1:4], g[4]
        val, mota = mota_RRR(rrr)
        khoang = BANG_tR.get(tr, "?")
        return f"[Đoạn 3] Lượng giáng thủy trong {khoang} qua: {mota}"
    if d0 == '7' and len(g) == 5:
        r24 = g[1:5]
        if r24 == "0000": return "Lượng mưa 24 giờ trước: không có mưa"
        if r24 == "9999": return "Lượng mưa 24 giờ trước: dạng giọt (< 0,1mm)"
        if r24 == "////": return "Lượng mưa 24 giờ trước: không đo được"
        return f"Lượng mưa 24 giờ trước (tính đến bản tin 12 GQT) = {int(r24)/10:.1f} mm"
    if d0 == '8' and len(g) == 5:
        ns, c, hs = g[1], g[2], g[3:5]
        if ns == '9' and g[2] == '/':
            return f"Bầu trời bị che khuất; tầm nhìn thẳng đứng ≈ {mota_hshs(g[3:5])}"
        return (f"Lớp/khối mây: lượng Ns={ns} ({BANG_N.get(ns,'?')}), "
                f"loại C={c} ({BANG_C.get(c,'?')}), độ cao chân mây ≈ {mota_hshs(hs)}")
    if d0 == '9' and len(g) == 5:
        sub3 = g[0:3]
        if sub3 == '911':
            ff = g[3:5]
            if ff.isdigit():
                tom_tat["gio_manh_nhat_911"] = int(ff)
            return f"Nhóm 911ff: Tốc độ gió giật mạnh nhất tức thời (≥16 m/s) trong kỳ quan trắc = {ff} m/s"
        if sub3 == '915':
            dd = g[3:5]
            if dd.isdigit():
                tom_tat["huong_gio_manh_nhat_915"] = int(dd)
            return f"Nhóm 915dd: Hướng gió ứng với tốc độ gió giật mạnh nhất (911ff) = mã {dd} ({mota_dd(dd)})"
        if sub3 == '919':
            return "Nhóm 919MwDa: Báo hiện tượng vòi rồng/lốc bụi - chưa hỗ trợ giải mã chi tiết."
        if sub3 == '926':
            return "Nhóm 926S0i0: Báo hiện tượng sương muối/giáng thủy nhuốm màu - chưa hỗ trợ giải mã chi tiết."
        if sub3 == '939':
            nn = g[3:5]
            return f"Nhóm 939nn: Đường kính hạt mưa đá lớn nhất ≈ {nn} mm"
        if sub3 in ('960', '961'):
            return "Nhóm 9SpSpspsp bổ sung hiện tượng thời tiết (Bảng mã 4677/4687) - chưa hỗ trợ giải mã chi tiết."
    return None


# ------------------------------------------------------------------
# METAR / SPECI
# ------------------------------------------------------------------

WW_PHENOMENA = {
    "DZ": "mưa phùn", "RA": "mưa", "SN": "tuyết", "SG": "hạt tuyết", "IC": "kim băng",
    "PL": "mưa đá nhỏ", "GR": "mưa đá", "GS": "mưa đá nhỏ/tuyết viên", "UP": "giáng thủy chưa xác định",
    "BR": "sương mù nhẹ (mù)", "FG": "sương mù", "FU": "khói", "VA": "tro núi lửa",
    "DU": "bụi lan rộng", "SA": "cát", "HZ": "mù khô (haze)", "PO": "lốc bụi/cát nhỏ",
    "SQ": "gió giật (squall)", "FC": "vòi rồng/lốc xoáy", "SS": "bão cát", "DS": "bão bụi",
    "TS": "dông",
}
WW_DESCRIPTOR = {
    "MI": "mỏng/nông", "PR": "cục bộ", "BC": "từng mảng", "DR": "cuốn thấp", "BL": "cuốn cao",
    "SH": "mưa rào", "TS": "kèm dông", "FZ": "đóng băng",
}
WW_INTENSITY = {"-": "nhẹ", "+": "mạnh", "VC": "lân cận sân bay"}


def _giai_ma_ww(code):
    m = re.match(r"^(-|\+|VC)?((?:MI|PR|BC|DR|BL|SH|TS|FZ){0,2})((?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS){1,3})$", code)
    if not m:
        return None
    intensity, desc, phen = m.groups()
    mo_ta_chinh = []
    if desc:
        for i in range(0, len(desc), 2):
            mo_ta_chinh.append(WW_DESCRIPTOR.get(desc[i:i+2], desc[i:i+2]))
    for i in range(0, len(phen), 2):
        mo_ta_chinh.append(WW_PHENOMENA.get(phen[i:i+2], phen[i:i+2]))
    cau = " ".join(mo_ta_chinh)
    if intensity in ("-", "+"):
        cau = f"{cau} ({WW_INTENSITY[intensity]})"
    elif intensity == "VC":
        cau = f"{cau} (ở lân cận sân bay)"
    return cau


# Thứ tự & tên cột chính xác theo file mẫu Excel do người dùng cung cấp
COT_MAU_EXCEL = [
    "Trạm", "Ngày", "giờ", "Nhiệt độ", "Điểm sương",
    "Khí áp mực biển (hPa)", "Khí áp mực trạm (hPa)", "Hướng gió ",
    "Tốc độ gió (m/s)", "Tầm nhìn (km)", "Mây tổng quan", "Mưa (mm)",
    "Xu thế khí áp", "Gió mạnh nhất trong kỳ quan trắc (mã 911)",
    "Hướng gió mạnh nhất trong kỳ quan trắc (mã 915)",
]


def synop_row_for_template(tom_tat):
    """Chuyển tom_tat (kết quả decode_synop) sang 1 dòng dữ liệu đúng theo mẫu Excel."""
    return {
        "Trạm": tom_tat.get("tram_ten"),
        "Ngày": tom_tat.get("ngay_so"),
        "giờ": tom_tat.get("gio_so"),
        "Nhiệt độ": tom_tat.get("nhiet_do_so"),
        "Điểm sương": tom_tat.get("diem_suong_so"),
        "Khí áp mực biển (hPa)": tom_tat.get("khi_ap_mbien_so"),
        "Khí áp mực trạm (hPa)": tom_tat.get("khi_ap_tram_so"),
        "Hướng gió ": tom_tat.get("huong_gio_so"),
        "Tốc độ gió (m/s)": tom_tat.get("toc_do_gio_so"),
        "Tầm nhìn (km)": tom_tat.get("tam_nhin_so"),
        "Mây tổng quan": tom_tat.get("may_tong_quan_ngan"),
        "Mưa (mm)": tom_tat.get("mua_so"),
        "Xu thế khí áp": tom_tat.get("xu_the_khi_ap_ngan"),
        "Gió mạnh nhất trong kỳ quan trắc (mã 911)": tom_tat.get("gio_manh_nhat_911"),
        "Hướng gió mạnh nhất trong kỳ quan trắc (mã 915)": tom_tat.get("huong_gio_manh_nhat_915"),
    }


def xuat_excel_synop(danh_sach_dong):
    """Xuất danh sách các dòng (dict theo COT_MAU_EXCEL) thành file Excel (bytes),
    giữ đúng tên cột như file mẫu; cột 'Mây tổng quan' được định dạng dạng văn bản
    để tránh Excel tự đổi '10/10' thành ngày tháng."""
    df_xuat = pd.DataFrame(danh_sach_dong, columns=COT_MAU_EXCEL)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_xuat.to_excel(writer, index=False, sheet_name="Sheet1")
        ws = writer.sheets["Sheet1"]
        col_idx = COT_MAU_EXCEL.index("Mây tổng quan") + 1
        for row in range(2, len(danh_sach_dong) + 2):
            ws.cell(row=row, column=col_idx).number_format = "@"
    buf.seek(0)
    return buf.getvalue()


def decode_metar(raw):
    text = raw.strip()
    text = text.rstrip("=")
    tokens = [t for t in re.split(r"\s+", text) if t]
    if not tokens:
        return {}, [], ["Bản tin trống."]

    chi_tiet = []
    ghi_chu = []
    tom_tat = {}
    idx = 0

    loai = tokens[idx]
    if loai in ("METAR", "SPECI"):
        chi_tiet.append((loai, "Bản tin thời tiết sân bay định kỳ (METAR)" if loai == "METAR"
                          else "Bản tin thời tiết sân bay đặc biệt chọn lọc (SPECI)"))
        idx += 1
    else:
        ghi_chu.append("Không thấy chỉ báo METAR/SPECI ở đầu bản tin, vẫn thử giải mã phần còn lại.")

    # Trạm
    if idx < len(tokens) and re.match(r"^[A-Z]{4}$", tokens[idx]):
        icao = tokens[idx]; idx += 1
        ten = VN_AIRPORTS.get(icao)
        if ten:
            chi_tiet.append((icao, f"Sân bay: {ten} (chỉ số ICAO {icao})"))
            tom_tat["san_bay"] = f"{ten} ({icao})"
        else:
            chi_tiet.append((icao, f"Chỉ số ICAO của sân bay: {icao}"))
            tom_tat["san_bay"] = icao

    # AUTO
    if idx < len(tokens) and tokens[idx] == "AUTO":
        chi_tiet.append(("AUTO", "Bản tin từ trạm quan trắc tự động, không có người can thiệp"))
        idx += 1

    # Ngày giờ
    if idx < len(tokens) and re.match(r"^\d{6}Z$", tokens[idx]):
        g = tokens[idx]; idx += 1
        dd, hh, mm = g[0:2], g[2:4], g[4:6]
        chi_tiet.append((g, f"Ngày {dd}, giờ quan trắc {hh}:{mm} UTC"))
        tom_tat["ngay"] = dd
        tom_tat["gio_utc"] = f"{hh}:{mm}"

    # Gió
    if idx < len(tokens):
        m = re.match(r"^(VRB|\d{3})(\d{2,3})(?:G(\d{2,3}))?(KT|MPS|KMH)$", tokens[idx])
        if m:
            g = tokens[idx]; idx += 1
            ddd, ff, gg, unit = m.groups()
            if ddd == "VRB":
                huong = "biến đổi"
            elif ddd == "000" and int(ff) == 0:
                huong = None
            else:
                huong = f"{ddd}°"
            if huong is None:
                mota = "Lặng gió (không có gió)"
            else:
                mota = f"Hướng gió {huong}, tốc độ {int(ff)} {unit}"
            if gg: mota += f", giật {int(gg)} {unit}"
            chi_tiet.append((g, mota))
            tom_tat["gio"] = mota

    # Hướng gió biến đổi (dnVdx)
    if idx < len(tokens) and re.match(r"^\d{3}V\d{3}$", tokens[idx]):
        g = tokens[idx]; idx += 1
        d1, d2 = g.split("V")
        chi_tiet.append((g, f"Hướng gió dao động giữa {d1}° và {d2}°"))

    # Tầm nhìn hoặc CAVOK
    if idx < len(tokens):
        if tokens[idx] == "CAVOK":
            g = tokens[idx]; idx += 1
            chi_tiet.append((g, "CAVOK: Tầm nhìn ≥10km, không mây dưới 1500m (không Cb), "
                                 "không hiện tượng thời tiết đáng chú ý"))
            tom_tat["tam_nhin"] = "≥ 10 km (CAVOK)"
        elif re.match(r"^\d{4}$", tokens[idx]):
            g = tokens[idx]; idx += 1
            if g == "9999":
                mota = "≥ 10 km"
            else:
                mota = f"{int(g)} m"
            chi_tiet.append((g, f"Tầm nhìn ngang phổ biến: {mota}"))
            tom_tat["tam_nhin"] = mota

    # RVR (bỏ qua chi tiết, chỉ ghi nhận)
    while idx < len(tokens) and re.match(r"^R\d{2}[LCR]?/", tokens[idx]):
        g = tokens[idx]; idx += 1
        chi_tiet.append((g, "Tầm nhìn đường băng (RVR) - chưa hỗ trợ giải mã chi tiết"))

    # Hiện tượng thời tiết (0-3 nhóm)
    while idx < len(tokens):
        mota_ww = _giai_ma_ww(tokens[idx])
        if mota_ww is None:
            break
        g = tokens[idx]; idx += 1
        chi_tiet.append((g, f"Hiện tượng thời tiết: {mota_ww}"))
        tom_tat.setdefault("hien_tuong_thoi_tiet", []).append(mota_ww)

    # Mây
    while idx < len(tokens):
        g = tokens[idx]
        if g in ("SKC", "NSC", "NCD", "CLR"):
            idx += 1
            mota = {"SKC": "Trời quang (Sky Clear)", "NSC": "Không có mây đáng chú ý (NSC)",
                    "NCD": "Không quan trắc được mây (trạm tự động, NCD)",
                    "CLR": "Trời quang (CLR)"}[g]
            chi_tiet.append((g, mota))
            tom_tat.setdefault("may", []).append(mota)
            continue
        m = re.match(r"^(FEW|SCT|BKN|OVC)(\d{3})(CB|TCU)?$", g)
        if m:
            idx += 1
            amt, hgt, extra = m.groups()
            amt_desc = {"FEW": "ít (1-2/8)", "SCT": "rải rác (3-4/8)",
                        "BKN": "nhiều (5-7/8)", "OVC": "phủ kín (8/8)"}[amt]
            do_cao = int(hgt) * 100
            extra_desc = ""
            if extra == "CB": extra_desc = " - mây vũ tích (Cb)"
            elif extra == "TCU": extra_desc = " - mây tích phát triển mạnh (TCU)"
            mota = f"Mây {amt_desc} ở độ cao chân mây ~{do_cao} ft{extra_desc}"
            chi_tiet.append((g, mota))
            tom_tat.setdefault("may", []).append(mota)
            continue
        m = re.match(r"^VV(\d{3}|///)$", g)
        if m:
            idx += 1
            hs = m.group(1)
            mota = f"Tầm nhìn thẳng đứng ~{int(hs)*100} ft (trời bị che khuất)" if hs != "///" else "Tầm nhìn thẳng đứng: không xác định"
            chi_tiet.append((g, mota))
            continue
        break

    # Nhiệt độ / điểm sương
    if idx < len(tokens):
        m = re.match(r"^(M?\d{2})/(M?\d{2})$", tokens[idx])
        if m:
            g = tokens[idx]; idx += 1
            t_raw, td_raw = m.groups()
            t = -int(t_raw[1:]) if t_raw.startswith("M") else int(t_raw)
            td = -int(td_raw[1:]) if td_raw.startswith("M") else int(td_raw)
            chi_tiet.append((g, f"Nhiệt độ không khí = {t}°C, nhiệt độ điểm sương = {td}°C"))
            tom_tat["nhiet_do"] = f"{t}°C"
            tom_tat["diem_suong"] = f"{td}°C"

    # Khí áp QNH
    if idx < len(tokens):
        m = re.match(r"^Q(\d{4})$", tokens[idx])
        if m:
            g = tokens[idx]; idx += 1
            chi_tiet.append((g, f"Khí áp QNH = {int(m.group(1))} hPa"))
            tom_tat["khi_ap_qnh"] = f"{int(m.group(1))} hPa"
        else:
            m = re.match(r"^A(\d{4})$", tokens[idx])
            if m:
                g = tokens[idx]; idx += 1
                val = int(m.group(1)) / 100.0
                chi_tiet.append((g, f"Khí áp QNH = {val:.2f} inHg"))
                tom_tat["khi_ap_qnh"] = f"{val:.2f} inHg"

    # Các phần còn lại: RExx, xu thế (NOSIG/BECMG/TEMPO), RMK...
    con_lai = tokens[idx:]
    if con_lai:
        rest_str = " ".join(con_lai)
        if "NOSIG" in con_lai:
            chi_tiet.append(("NOSIG", "Không dự báo có thay đổi đáng kể trong 2 giờ tới"))
            tom_tat["xu_the"] = "Không đổi (NOSIG)"
            con_lai.remove("NOSIG")
        if con_lai and con_lai[0] in ("BECMG", "TEMPO"):
            chi_tiet.append((" ".join(con_lai), f"Dự báo xu thế ({con_lai[0]}) - chi tiết cần tra cứu thêm"))
            tom_tat["xu_the"] = f"{con_lai[0]} (xem nhóm gốc)"
            con_lai = []
        if con_lai:
            chi_tiet.append((" ".join(con_lai), "Phần còn lại của bản tin - chưa hỗ trợ giải mã chi tiết"))

    return tom_tat, chi_tiet, ghi_chu


# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    if 'interpol_fig' not in st.session_state: st.session_state['interpol_fig'] = None
    if 'folium_map_obj' not in st.session_state: st.session_state['folium_map_obj'] = None
    if 'folium_fig_obj' not in st.session_state: st.session_state['folium_fig_obj'] = None
    if 'interp_cache' not in st.session_state: st.session_state['interp_cache'] = None
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'storm_fig_obj' not in st.session_state: st.session_state['storm_fig_obj'] = None

    with st.sidebar:
        st.title("Dữ liệu thời tiết")
        topic = st.radio("CHỌN CHẾ ĐỘ:", ["Bản đồ Bão", "Ảnh mây vệ tinh", "Dữ liệu quan trắc", "Dự báo điểm (KMA)"])
        st.markdown("---")

        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""
        obs_mode = ""

        title_interpol = ""
        data_file_interpol = None
        btn_run_interpol = False
        custom_bounds_dict = None

        # ---- Biến mới cho Dịch mã điện ----
        decode_input_text = ""
        btn_run_decode = False

        use_storm_bounds = False
        storm_bounds_dict = None
        show_grid = False

        # ---- Biến mới cho lớp xã ----
        show_xa_layer  = False
        show_xa_labels = False
        nc_var_selected = None
        nc_time_idx = None
        cmap_option = 'jet'
        num_bins = 10
        custom_levels = None
        selected_provinces = []
        shape_col = ""

        if topic == "Dữ liệu quan trắc":
            if st.session_state['logged_in']:
                obs_mode = st.radio("Chọn nguồn dữ liệu:", ["Thời tiết (WeatherObs)", "Gió tự động (KTTV)", "Nội suy nhiệt độ", "Nội suy lượng mưa", "Nội suy linh tinh", "Dịch mã điện"])

                if obs_mode in ["Nội suy nhiệt độ", "Nội suy lượng mưa"]:
                    st.markdown("---")
                    st.markdown(f"### 🛠️ CÔNG CỤ {obs_mode.upper()}")
                    default_title = "Bản đồ nhiệt độ nội suy" if obs_mode == "Nội suy nhiệt độ" else "Bản đồ lượng mưa nội suy"
                    title_interpol = st.text_input("Tiêu đề bản đồ:", value=default_title)
                    st.markdown("**1. Upload dữ liệu (.xlsx/.csv)**")
                    st.caption("Cột: `stations`, `lon`, `lat`, `value`")
                    data_file_interpol = st.file_uploader("Chọn file số liệu:", type=['xlsx', 'csv'], key="data_up")
                    st.markdown("---")
                    btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ", type="primary", use_container_width=True)

                elif obs_mode == "Nội suy linh tinh":
                    st.markdown("---")
                    st.markdown("### 🛠️ NỘI SUY TÙY BIẾN (TƯƠNG TÁC)")
                    title_interpol = st.text_input("Tiêu đề bản đồ:", value="Bản đồ Nội Suy Tùy Chọn", key="title_custom_interp")
                    data_file_interpol = st.file_uploader("Chọn file số liệu:", type=['xlsx', 'csv', 'nc'], key="data_up_custom")

                    if data_file_interpol and data_file_interpol.name.endswith('.nc'):
                        try:
                            tmp_path = "sidebar_check.nc"
                            with open(tmp_path, "wb") as f: f.write(data_file_interpol.getvalue())
                            ds_tmp = None
                            for eng in ['netcdf4', 'scipy', 'h5netcdf', None]:
                                try:
                                    ds_tmp = xr.open_dataset(tmp_path, engine=eng)
                                    break
                                except: pass

                            if ds_tmp is None:
                                st.error("Không thể đọc định dạng NetCDF này. Vui lòng kiểm tra lại file của bạn.")
                            else:
                                vars_list = list(ds_tmp.data_vars.keys())
                                if vars_list: nc_var_selected = st.selectbox("📌 Chọn biến dữ liệu (Variable):", vars_list)
                                time_dim = next((d for d in ds_tmp.dims if d.lower() in ['time', 't', 'valid_time']), None)

                                if time_dim:
                                    time_values = ds_tmp[time_dim].values
                                    if np.issubdtype(time_values.dtype, np.datetime64):
                                        time_options = [pd.to_datetime(str(t)).strftime("%Y-%m-%d %H:%M:%S") for t in time_values]
                                    else:
                                        time_options = [str(t) for t in time_values]
                                    selected_time_str = st.selectbox("⏳ Chọn thời gian (Time Step):", time_options)
                                    nc_time_idx = time_options.index(selected_time_str)
                                else: st.info("File NetCDF không có dimension thời gian (Time).")
                                ds_tmp.close()

                            if os.path.exists(tmp_path):
                                try: os.remove(tmp_path)
                                except: pass
                            data_file_interpol.seek(0)
                        except Exception as e: st.error(f"Lỗi đọc file NetCDF: {e}")

                    st.markdown("**1. Cấu hình màu & Ngưỡng**")
                    cmap_list = plt.colormaps()
                    default_cmap_idx = cmap_list.index('jet') if 'jet' in cmap_list else 0
                    cmap_option = st.selectbox("Chọn thang màu (Colormap):", cmap_list, index=default_cmap_idx)

                    fig_cmap, ax_cmap = plt.subplots(figsize=(3, 0.2))
                    fig_cmap.subplots_adjust(top=1, bottom=0, left=0, right=1)
                    gradient = np.linspace(0, 1, 256).reshape(1, -1)
                    ax_cmap.imshow(gradient, aspect='auto', cmap=cmap_option)
                    ax_cmap.set_axis_off()
                    st.pyplot(fig_cmap)

                    threshold_type = st.radio("Cách chia ngưỡng:", ["Tự động (Số lớp)", "Tùy chỉnh (Nhập tay)"])
                    num_bins = 10
                    custom_levels = None
                    if threshold_type == "Tự động (Số lớp)":
                        num_bins = st.number_input("Số lượng ngưỡng chia:", min_value=2, max_value=50, value=10)
                    else:
                        custom_levels_str = st.text_input("Nhập các ngưỡng (cách nhau bằng dấu phẩy):", "0, 10, 20, 30, 40, 50")
                        try: custom_levels = [float(x.strip()) for x in custom_levels_str.split(',') if x.strip()]
                        except: st.error("Lỗi định dạng. Vui lòng nhập số.")

                    st.markdown("**2. Ranh giới Tỉnh**")
                    province_list = []
                    shape_col = ""
                    if os.path.exists(SHP_MASK_PATH):
                        try:
                            tmp_shp = gpd.read_file(SHP_MASK_PATH)
                            for col in ['TEN_TINH', 'NAME_1', 'Name', 'PROVINCE', 'Tỉnh', 'Tinh', 'tentinh', 'Ten_Tinh']:
                                if col in tmp_shp.columns:
                                    shape_col = col
                                    province_list = sorted(tmp_shp[col].dropna().unique().tolist())
                                    break
                        except: pass

                    selected_provinces = []
                    if province_list:
                        quick_prov = st.selectbox("Hộp chọn nhanh 1 Tỉnh (Bản đồ chính):", ["-- Tất cả Tỉnh --"] + province_list)
                        multi_provs = st.multiselect("Hoặc chọn thủ công nhiều Tỉnh:", province_list)
                        selected_provinces = [quick_prov] if quick_prov != "-- Tất cả Tỉnh --" else multi_provs

                    # ============================================================
                    # MỤC 3 – LỚP RANH GIỚI XÃ (MỚI)
                    # ============================================================
                    st.markdown("**3. Lớp ranh giới Xã**")
                    xa_shp_exists = os.path.exists(SHP_XA_PATH)
                    if not xa_shp_exists:
                        st.caption("⚠️ Chưa tìm thấy `shp/RG_xa_VN.shp`.")
                    show_xa_layer = st.checkbox(
                        "🏘️ Hiển thị ranh giới Xã",
                        value=False,
                        disabled=not xa_shp_exists,
                        help="Tải và vẽ lớp ranh giới xã từ shp/RG_xa_VN.shp"
                    )
                    if show_xa_layer and xa_shp_exists:
                        show_xa_labels = st.checkbox(
                            "🏷️ Hiển thị tên Xã (ten_xa)",
                            value=False,
                            help="Hiển thị nhãn tên xã trên bản đồ (có thể làm chậm nếu nhiều xã)"
                        )

                    # ============================================================
                    # MỤC 4 – CẮT CÚP TỌA ĐỘ
                    # ============================================================
                    st.markdown("**4. Cắt cúp theo Tọa độ**")
                    use_custom_bounds = st.checkbox("✂️ Giới hạn tải & hiển thị theo Tọa độ", value=False)
                    if use_custom_bounds:
                        col_b1, col_b2 = st.columns(2)
                        with col_b1: min_lon = st.number_input("Kinh độ Min", value=101.80); min_lat = st.number_input("Vĩ độ Min", value=8.00)
                        with col_b2: max_lon = st.number_input("Kinh độ Max", value=115.00); max_lat = st.number_input("Vĩ độ Max", value=24.00)
                        custom_bounds_dict = {'minx': min_lon, 'maxx': max_lon, 'miny': min_lat, 'maxy': max_lat}

                    st.markdown("---")
                    btn_run_interpol = st.button("🚀 VẼ BẢN ĐỒ TƯƠNG TÁC", type="primary", use_container_width=True)

                elif obs_mode == "Dịch mã điện":
                    st.markdown("---")
                    st.markdown("### 🛠️ DỊCH MÃ ĐIỆN (SYNOP / METAR)")
                    st.caption("Dán một hoặc nhiều bản tin SYNOP (AAXX...) hoặc METAR/SPECI, mỗi bản tin một dòng.")
                    decode_input_text = st.text_area(
                        "Nội dung bản tin:",
                        value="AAXX 06001 48823 12497 71602 10287 20247 30017 40020 53005 60022 85808 333 01028 20275 3/026 59002 82894 85696\nMETAR VVDB 060200Z VRB02KT 9999 FEW007 SCT024 OVC030 25/24 Q1007 NOSIG=",
                        height=180,
                        key="decode_input_text"
                    )
                    btn_run_decode = st.button("🔎 GIẢI MÃ", type="primary", use_container_width=True)

                st.markdown("---")
                if st.button("🔒 Đăng xuất", key="logout_obs_sidebar"):
                    st.session_state['logged_in'] = False
                    st.rerun()

        if topic == "Dự báo điểm (KMA)":
            if st.session_state['logged_in']:
                st.markdown("---")
                if st.button("🔒 Đăng xuất", key="logout_kma_sidebar"):
                    st.session_state['logged_in'] = False
                    st.rerun()

        if topic == "Bản đồ Bão":
            storm_opt = st.selectbox("Dữ liệu bão:", ["Hiện trạng (Besttrack)", "Lịch sử (Historical)"])
            default_title = "TIN BÃO KHẨN CẤP" if "Hiện trạng" in storm_opt else "THỐNG KÊ LỊCH SỬ"
            dashboard_title = st.text_input("Tiêu đề bảng thông tin:", value=default_title)
            active_mode = storm_opt

            show_grid = st.checkbox("🌐 Hiển thị Lưới tọa độ nền", value=True)

            if "Hiện trạng" in storm_opt:
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack (.csv / .xlsx)", type=["csv", "xlsx"], key="o1")
                    if f:
                        try:
                            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                            df = normalize_columns(df)
                            if 'name' not in df: df['name'], df['storm_no'] = 'Storm', 'Current'
                            for c in ['wind_km/h','bf','r6','r10','rc','pressure','hour_explicit']:
                                if c not in df: df[c]=0
                            df = df.dropna(subset=['lat','lon'])
                            all_s = df['storm_no'].unique() if 'storm_no' in df else []
                            sel = st.multiselect("Chọn cơn bão:", all_s, default=all_s) if len(all_s)>0 else []
                            final_df = df[df['storm_no'].isin(sel)] if len(sel)>0 else df
                        except: pass
                    else: st.info("Vui lòng upload file dữ liệu để xem thông tin bão.")

                    st.markdown("**Cắt cúp theo Tọa độ (Trích xuất ảnh tĩnh)**")
                    use_storm_bounds = st.checkbox("✂️ Giới hạn tọa độ vẽ biểu đồ", value=False, key="storm_bounds_chk")
                    if use_storm_bounds:
                        col_s1, col_s2 = st.columns(2)
                        with col_s1: s_min_lon = st.number_input("Kinh độ Min (Bão)", value=100.0); s_min_lat = st.number_input("Vĩ độ Min (Bão)", value=5.0)
                        with col_s2: s_max_lon = st.number_input("Kinh độ Max (Bão)", value=125.0); s_max_lat = st.number_input("Vĩ độ Max (Bão)", value=25.0)
                        storm_bounds_dict = {'minx': s_min_lon, 'maxx': s_max_lon, 'miny': s_min_lat, 'maxy': s_max_lat}

            else:
                if st.checkbox("Hiển thị lớp Dữ liệu", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack_capgio.xlsx", type="xlsx", key="o2")
                    if f:
                        try:
                            df = pd.read_excel(f)
                            df = normalize_columns(df)
                            df = df.dropna(subset=['lat','lon'])
                            years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                            temp = df[df['year'].isin(years)]
                            names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                            final_df = temp[temp['name'].isin(names)]
                        except: pass
                    else: st.info("Vui lòng upload file dữ liệu lịch sử bão.")

                    st.markdown("**Cắt cúp theo Tọa độ (Trích xuất ảnh tĩnh)**")
                    use_storm_bounds = st.checkbox("✂️ Giới hạn tọa độ vẽ biểu đồ", value=False, key="storm_bounds_chk_his")
                    if use_storm_bounds:
                        col_s1, col_s2 = st.columns(2)
                        with col_s1: s_min_lon = st.number_input("Kinh độ Min (Bão)", value=100.0); s_min_lat = st.number_input("Vĩ độ Min (Bão)", value=5.0)
                        with col_s2: s_max_lon = st.number_input("Kinh độ Max (Bão)", value=125.0); s_max_lat = st.number_input("Vĩ độ Max (Bão)", value=25.0)
                        storm_bounds_dict = {'minx': s_min_lon, 'maxx': s_max_lon, 'miny': s_min_lat, 'maxy': s_max_lat}

    # --- MAIN CONTENT ---
    if topic == "Ảnh mây vệ tinh":
        components.iframe("https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=1000&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1")

    elif topic == "Dữ liệu quan trắc":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập Hệ thống")
            st.info("Vui lòng đăng nhập để truy cập Dữ liệu Quan trắc & Dự báo KMA.")
            with st.form("login_form_common"):
                user_input = st.text_input("Tên đăng nhập")
                pass_input = st.text_input("Mật khẩu", type="password")
                if st.form_submit_button("Đăng nhập"):
                    if user_input == "admin" and pass_input == "kttv@2026":
                        st.session_state['logged_in'] = True
                        st.success("Đăng nhập thành công!")
                        st.rerun()
                    else: st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
        else:
            if "WeatherObs" in obs_mode:
                st.markdown(f'<div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;"><iframe src="{LINK_WEATHEROBS}" style="width: calc(100% + 19px); height: 1000px; position: absolute; top: -50px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
            elif "Gió tự động" in obs_mode:
                 st.markdown(f'<div style="overflow: hidden; width: 100%; height: 95vh; position: relative; border: 1px solid #ddd;"><iframe src="{LINK_WIND_AUTO}" style="width: calc(100% + 19px); height: 1200px; position: absolute; top: -75px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
            elif obs_mode in ["Nội suy nhiệt độ", "Nội suy lượng mưa"]:
                if btn_run_interpol:
                    if data_file_interpol:
                        try:
                            df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)
                            data_type = 'rain' if obs_mode == "Nội suy lượng mưa" else 'temp'
                            with st.spinner("Đang tính toán nội suy và tạo bản đồ..."):
                                fig, err = run_interpolation_and_plot(df_in, title_interpol, data_type)
                                if err: st.error(f"❌ {err}")
                                else: st.session_state['interpol_fig'] = fig
                        except Exception as e: st.error(f"❌ Lỗi: {e}")
                    else: st.toast("Vui lòng upload file dữ liệu trước!", icon="⚠️")

                if st.session_state['interpol_fig']:
                    st.pyplot(st.session_state['interpol_fig'], use_container_width=True)
                    st.markdown("### 📥 Tải xuống")
                    col_dl1, col_dl2 = st.columns([1, 3])
                    with col_dl1: fmt = st.selectbox("Định dạng:", ["png", "pdf"], key="fmt_static")
                    buf = io.BytesIO()
                    st.session_state['interpol_fig'].savefig(buf, format=fmt, dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    with col_dl2:
                        st.write(""); st.write("")
                        st.download_button(label=f"⬇️ Tải ảnh về ({fmt.upper()})", data=buf.getvalue(), file_name=f"ban_do_tinh.{fmt}", mime=f"image/{fmt}" if fmt=="png" else "application/pdf", key="dl_btn_static")
                else: st.info("👈 Vui lòng cấu hình và nhấn nút 'VẼ BẢN ĐỒ' ở thanh menu bên trái.")

            elif obs_mode == "Nội suy linh tinh":
                if btn_run_interpol:
                    if data_file_interpol:
                        try:
                            if data_file_interpol.name.endswith('.nc'):
                                tmp_path = "run_check.nc"
                                with open(tmp_path, "wb") as f: f.write(data_file_interpol.getvalue())

                                ds = None
                                for eng in ['netcdf4', 'scipy', 'h5netcdf', None]:
                                    try:
                                        ds = xr.open_dataset(tmp_path, engine=eng)
                                        break
                                    except: pass

                                if ds is None:
                                    st.error("Không thể đọc file NetCDF khi vẽ. Vui lòng đảm bảo file không bị hỏng.")
                                    df_in = pd.DataFrame()
                                else:
                                    time_dim = next((d for d in ds.dims if d.lower() in ['time', 't', 'valid_time']), None)
                                    if time_dim and nc_time_idx is not None: ds = ds.isel({time_dim: nc_time_idx})

                                    var_name = nc_var_selected if nc_var_selected else list(ds.data_vars.keys())[0]
                                    df_nc = ds[var_name].to_dataframe().reset_index()

                                    lat_col = next((c for c in df_nc.columns if c.lower() in ['lat', 'latitude', 'y']), None)
                                    lon_col = next((c for c in df_nc.columns if c.lower() in ['lon', 'longitude', 'x']), None)

                                    if lat_col and lon_col:
                                        df_in = df_nc.rename(columns={lat_col: 'lat', lon_col: 'lon', var_name: 'value'})
                                        df_in = df_in[['lon', 'lat', 'value']].dropna()
                                    else:
                                        st.error("Không tìm thấy các biến tọa độ lat/lon thông dụng trong file NetCDF.")
                                        df_in = pd.DataFrame()
                                    ds.close()

                                if os.path.exists(tmp_path):
                                    try: os.remove(tmp_path)
                                    except: pass
                                data_file_interpol.seek(0)
                            else:
                                df_in = pd.read_csv(data_file_interpol) if data_file_interpol.name.endswith('.csv') else pd.read_excel(data_file_interpol)

                            if not df_in.empty:
                                with st.spinner("Đang xử lý nội suy tương tác và trích xuất bản vẽ..."):
                                    m_map, m_fig, m_cache, err = run_interactive_folium_interpolation(
                                        df_in, title_interpol, cmap_option, num_bins, custom_levels,
                                        selected_provinces, shape_col, custom_bounds_dict,
                                        show_xa_layer=show_xa_layer,
                                        show_xa_labels=show_xa_labels
                                    )
                                    if err: st.error(f"❌ Lỗi: {err}")
                                    else:
                                        st.session_state['folium_map_obj'] = m_map
                                        st.session_state['folium_fig_obj'] = m_fig
                                        st.session_state['interp_cache'] = m_cache
                        except Exception as e: st.error(f"❌ Lỗi Xử lý Dữ liệu: {e}")
                    else: st.toast("Vui lòng upload file dữ liệu trước!", icon="⚠️")

                if st.session_state['folium_map_obj']:
                    st.success("Bản đồ thành công! Kéo xuống để tải Toàn bộ khu vực, HOẶC dùng hộp chọn/click bản đồ để tải riêng từng Tỉnh.")
                    map_data = st_folium(st.session_state['folium_map_obj'], width=None, height=800, use_container_width=True, returned_objects=["last_active_drawing"])

                    st.markdown("---")
                    st.markdown("### 📥 Tải bản vẽ tĩnh (Toàn bộ khu vực đã chọn)")
                    col_dl1, col_dl2 = st.columns([1, 3])
                    with col_dl1: fmt = st.selectbox("Định dạng:", ["png", "pdf"], key="fmt_folium")
                    buf = io.BytesIO()
                    st.session_state['folium_fig_obj'].savefig(buf, format=fmt, dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    with col_dl2:
                        st.write(""); st.write("")
                        st.download_button(label=f"⬇️ Tải Toàn bộ ({fmt.upper()})", data=buf.getvalue(), file_name=f"ban_do_tong.{fmt}", mime=f"image/{fmt}" if fmt=="png" else "application/pdf", key="dl_btn_folium")

                    cache = st.session_state.get('interp_cache')
                    if cache and cache.get('mask_shape') is not None and not cache['mask_shape'].empty:
                        st.markdown("---")
                        st.markdown("### 🎯 Tải bản vẽ Tỉnh riêng lẻ (Cắt theo ranh giới tỉnh)")
                        shape_col_cache = cache.get('shape_col', "")

                        if not shape_col_cache or shape_col_cache not in cache['mask_shape'].columns:
                            st.warning("⚠️ Không thể tải riêng từng tỉnh vì Shapefile (vn34tinh.shp) không có cột chứa tên Tỉnh. Tính năng này tạm thời bị vô hiệu hóa.")
                        else:
                            available_provs = sorted(cache['mask_shape'][shape_col_cache].dropna().unique().tolist())
                            col_sel1, col_sel2 = st.columns([2, 2])
                            with col_sel1: selected_dl_prov = st.selectbox("Hộp chọn Tỉnh muốn tải:", ["-- Click trên bản đồ hoặc Chọn tại đây --"] + available_provs)

                            clicked_prov = None
                            if map_data and map_data.get("last_active_drawing"):
                                props = map_data["last_active_drawing"].get("properties", {})
                                if shape_col_cache in props:
                                    clicked_prov = props[shape_col_cache]
                                    with col_sel2:
                                        st.write(""); st.info(f"💡 Đang click chọn: **{clicked_prov}** trên bản đồ.")

                            final_dl_prov = selected_dl_prov if selected_dl_prov != "-- Click trên bản đồ hoặc Chọn tại đây --" else clicked_prov
                            if final_dl_prov:
                                prov_fig = generate_single_province_fig(cache, final_dl_prov, st.session_state.get("title_custom_interp", "Bản đồ Nội Suy"))
                                if prov_fig:
                                    col_p1, col_p2 = st.columns([1, 3])
                                    with col_p1: fmt_prov = st.selectbox("Định dạng ảnh:", ["png", "pdf"], key="fmt_prov")
                                    buf_prov = io.BytesIO()
                                    prov_fig.savefig(buf_prov, format=fmt_prov, dpi=100, bbox_inches='tight')
                                    buf_prov.seek(0)
                                    with col_p2:
                                        st.write(""); st.write("")
                                        st.download_button(label=f"⬇️ Tải ảnh Tỉnh {final_dl_prov} ({fmt_prov.upper()})", data=buf_prov.getvalue(), file_name=f"ban_do_{final_dl_prov}.{fmt_prov}", mime=f"image/{fmt_prov}" if fmt_prov=="png" else "application/pdf", key="dl_btn_prov")
                else: st.info("👈 Vui lòng cấu hình dữ liệu, chọn màu, ngưỡng, tọa độ và nhấn 'VẼ BẢN ĐỒ TƯƠNG TÁC'.")

            elif obs_mode == "Dịch mã điện":
                if btn_run_decode:
                    raw_lines = [ln.strip() for ln in decode_input_text.splitlines() if ln.strip()]
                    danh_sach_dong_synop = []
                    for line in raw_lines:
                        first_word = line.split()[0] if line.split() else ""
                        if first_word in ("AAXX", "BBXX", "OOXX"):
                            tom_tat, _, _ = decode_synop(line)
                            if tom_tat:
                                danh_sach_dong_synop.append(synop_row_for_template(tom_tat))
                    if danh_sach_dong_synop:
                        df_mau = pd.DataFrame(danh_sach_dong_synop, columns=COT_MAU_EXCEL)
                        st.dataframe(df_mau, use_container_width=True, hide_index=True)
                        excel_bytes = xuat_excel_synop(danh_sach_dong_synop)
                        st.download_button(
                            label="⬇️ Tải bảng Excel (.xlsx)",
                            data=excel_bytes,
                            file_name="giai_ma_synop.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_btn_decode_excel"
                        )

    elif topic == "Dự báo điểm (KMA)":
        if not st.session_state['logged_in']:
            st.title("🔐 Đăng nhập Hệ thống")
            st.info("Vui lòng đăng nhập để truy cập Dữ liệu Quan trắc & Dự báo KMA.")
            with st.form("login_form_common_kma"):
                user_input = st.text_input("Tên đăng nhập")
                pass_input = st.text_input("Mật khẩu", type="password")
                if st.form_submit_button("Đăng nhập"):
                    if user_input == "admin" and pass_input == "kttv@2026":
                        st.session_state['logged_in'] = True
                        st.success("Đăng nhập thành công!")
                        st.rerun()
                    else: st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
        else:
            realtime_kma_url = get_kma_url()
            st.markdown(f'<div style="overflow: hidden; width: 100%; height: 700px; position: relative; border: 1px solid #ddd;"><iframe src="{realtime_kma_url}" style="width: calc(100% + 19px); height: 1200px; position: absolute; top: -130px; left: 0px; border: none;" allow="fullscreen"></iframe></div>', unsafe_allow_html=True)
            st.caption(f"Đang hiển thị dữ liệu từ nguồn KMA (Hàn Quốc). Link gốc: {realtime_kma_url}")

    elif topic == "Bản đồ Bão":
        start_loc = [16.0, 114.0]
        if use_storm_bounds and storm_bounds_dict:
            start_loc = [(storm_bounds_dict['miny'] + storm_bounds_dict['maxy'])/2, (storm_bounds_dict['minx'] + storm_bounds_dict['maxx'])/2]

        m = folium.Map(location=start_loc, zoom_start=6, tiles=None, zoom_control=False)
        if use_storm_bounds and storm_bounds_dict:
            m.fit_bounds([[storm_bounds_dict['miny'], storm_bounds_dict['minx']], [storm_bounds_dict['maxy'], storm_bounds_dict['maxx']]])

        storm_tile_layer = folium.TileLayer('CartoDB positron', name='Bản đồ Nền (Mặc định)', overlay=False, control=False, cross_origin=True)
        storm_tile_layer.add_to(m)

        ts = get_rainviewer_ts()
        if ts: folium.TileLayer(tiles=f"https://tile.rainviewer.com/{ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png", attr="RainViewer", name="☁️ Mây Vệ tinh", overlay=True, show=True, opacity=0.5).add_to(m)

        if show_grid:
            grid_fg = folium.FeatureGroup(name="🌐 Lưới Tọa độ", show=True)
            for lat in range(-85, 86, 5):
                folium.PolyLine([[lat, -180], [lat, 180]], color='gray', weight=1, dash_array='4, 4', opacity=0.4).add_to(grid_fg)
            for lon in range(-180, 181, 5):
                folium.PolyLine([[-85, lon], [85, lon]], color='gray', weight=1, dash_array='4, 4', opacity=0.4).add_to(grid_fg)
            grid_fg.add_to(m)

        fg_storm = folium.FeatureGroup(name="🌀 Đường đi Bão")
        if not final_df.empty and show_widgets:
            if "Hiện trạng" in str(active_mode):
                groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
                for g in groups:
                    sub = final_df[final_df['storm_no']==g] if g else final_df
                    dense = densify_track(sub)
                    f6, f10, fc = create_storm_swaths(dense)
                    for geom, c, o in [(f6,'#FFC0CB',0.4), (f10,'#FF6347',0.5), (fc,'#90EE90',0.6)]:
                        if geom and not geom.is_empty: folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg_storm)
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_storm)

                    for _, r in sub.iterrows():
                        icon_key = get_icon_name(r)
                        icon_path = ICON_PATHS.get(icon_key)
                        icon_base64 = image_to_base64(icon_path) if icon_path else None
                        if icon_base64:
                            i_size, i_anchor = ((20, 20), (10, 10)) if 'vungthap' in icon_key else ((40, 40), (20, 20))
                            icon = folium.CustomIcon(icon_image=icon_base64, icon_size=i_size, icon_anchor=i_anchor)
                            folium.Marker(location=[r['lat'], r['lon']], icon=icon, tooltip=f"Vmax {int(r.get('wind_km/h', 0))} km/h").add_to(fg_storm)
            else:
                for n in final_df['name'].unique():
                    sub = final_df[final_df['name']==n].sort_values('dt')
                    folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2).add_to(fg_storm)
                    for _, r in sub.iterrows():
                        c = '#00f2ff' if r.get('wind_km/h',0)<64 else '#ff0055'
                        folium.CircleMarker([r['lat'],r['lon']], radius=3, color=c, fill=True, popup=f"{n}").add_to(fg_storm)

        fg_storm.add_to(m)
        folium.LayerControl(position='topleft', collapsed=False).add_to(m)
        them_nut_chup_anh_ban_do(m, ten_file="ban_do_bao", ten_tile_layer=storm_tile_layer)

        if show_widgets:
            html_to_render = '<div class="floating-container">'
            if "Hiện trạng" in str(active_mode) and os.path.exists(CHUTHICH_IMG):
                with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                html_to_render += f'<div class="legend-box"><img src="data:image/png;base64,{b64}"></div>'
            html_to_render += create_info_table(final_df, dashboard_title) if not final_df.empty else create_info_table(pd.DataFrame(), "ĐANG TẢI DỮ LIỆU...")
            html_to_render += '</div>'
            st.markdown(html_to_render, unsafe_allow_html=True)

        st_folium(m, width=None, height=800, use_container_width=True)

        if show_widgets and not final_df.empty:
            st.markdown("---")
            st.markdown("### 📥 Tải bản đồ Bão (Ảnh tĩnh có chú thích và bảng tin y như hình gốc)")

            if st.button("🚀 Trích xuất Bản đồ Bão tĩnh", type="primary"):
                with st.spinner("Đang xây dựng bản vẽ tĩnh chuyên nghiệp..."):
                    st.session_state['storm_fig_obj'] = generate_storm_static_fig(final_df, dashboard_title, storm_bounds_dict if use_storm_bounds else None)

            if st.session_state['storm_fig_obj'] is not None:
                st.pyplot(st.session_state['storm_fig_obj'])

                col_dl1, col_dl2 = st.columns([1, 3])
                with col_dl1:
                    fmt_storm = st.selectbox("Định dạng:", ["png", "pdf"], key="fmt_storm")

                buf_storm = io.BytesIO()
                if fmt_storm == "png":
                    st.session_state['storm_fig_obj'].savefig(buf_storm, format="png", dpi=100, bbox_inches='tight')
                    mime_type = "image/png"
                else:
                    st.session_state['storm_fig_obj'].savefig(buf_storm, format="pdf", bbox_inches='tight')
                    mime_type = "application/pdf"

                buf_storm.seek(0)
                with col_dl2:
                    st.write(""); st.write("")
                    st.download_button(
                        label=f"⬇️ Tải bản đồ ({fmt_storm.upper()})",
                        data=buf_storm.getvalue(),
                        file_name=f"ban_do_bao_tinh.{fmt_storm}",
                        mime=mime_type,
                        key="dl_btn_storm"
                    )

if __name__ == "__main__":
    main()
