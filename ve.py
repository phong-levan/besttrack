# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
from math import radians, sin, cos, asin, sqrt
import warnings
import textwrap

# Thư viện xử lý hình học cho Option 1
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH HỆ THỐNG ---
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"        # File Hiện trạng/Dự báo
FILE_OPT2 = "besttrack_capgio.xlsx" # File Lịch sử
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(
    page_title="Hệ thống Giám sát Bão Biển Đông", 
    layout="wide", 
    initial_sidebar_state="expanded" # Mở sidebar để người dùng thấy 2 options
)

# --- 2. CSS "XUYÊN THẤU" (SỬA LỖI TRẮNG MÀN HÌNH) ---
st.markdown("""
    <style>
    /* Làm trong suốt toàn bộ nền ứng dụng Streamlit */
    .stApp { background: transparent !important; }
    [data-testid="stAppViewContainer"] { background: transparent !important; }
    [data-testid="stHeader"], footer { display: none !important; }
    
    /* Reset lề */
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    
    /* Bản đồ (iframe) nằm dưới cùng */
    iframe {
        position: fixed; top: 0; left: 0;
        width: 100vw !important; height: 100vh !important;
        z-index: 0;
    }
    
    /* Đảm bảo Sidebar và Dashboard nằm đè lên bản đồ */
    [data-testid="stSidebar"] { 
        z-index: 1001 !important; 
        background-color: rgba(28, 35, 49, 0.95) !important; /* Màu nền tối cho sidebar */
    }
    .dashboard-box { z-index: 1000 !important; }
    
    /* Style cho Radio Button (2 Ops chọn) */
    div[role="radiogroup"] > label {
        background: #262730;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        border: 1px solid #444;
        width: 100%;
    }
    div[role="radiogroup"] > label:hover {
        border-color: #00d4ff;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CÁC HÀM XỬ LÝ SỐ LIỆU ---

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def densify_track(df, step_km=10):
    new_rows = []
    if len(df) < 2: return df
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine_km(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(steps):
            f = j / steps
            row = p1.copy()
            row['lat'] = p1['lat'] + (p2['lat'] - p1['lat']) * f
            row['lon'] = p1['lon'] + (p2['lon'] - p1['lon']) * f
            for col in ['bán kính gió mạnh cấp 6 (km)', 'bán kính gió mạnh cấp 10 (km)', 'bán kính tâm (km)']:
                if col in p1 and col in p2:
                    row[col] = p1.get(col, 0)*(1-f) + p2.get(col, 0)*f
            new_rows.append(row)
    new_rows.append(df.iloc[-1])
    return pd.DataFrame(new_rows)

def create_storm_swaths(dense_df):
    polys = {'r6': [], 'r10': [], 'rc': []}
    geo = geodesic.Geodesic()
    for _, row in dense_df.iterrows():
        r6 = row.get('bán kính gió mạnh cấp 6 (km)', 0)
        r10 = row.get('bán kính gió mạnh cấp 10 (km)', 0)
        rc = row.get('bán kính tâm (km)', 0)
        for r, key in [(r6, 'r6'), (r10, 'r10'), (rc, 'rc')]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                polys[key].append(Polygon(circle))
    u = {k: unary_union(v) if v else None for k, v in polys.items()}
    f_rc = u['rc']
    f_r10 = u['r10'].difference(u['rc']) if u['r10'] and u['rc'] else u['r10']
    f_r6 = u['r6'].difference(u['r10']) if u['r6'] and u['r10'] else u['r6']
    return f_r6, f_r10, f_rc

def get_storm_icon(row):
    status = "daqua" if "quá khứ" in str(row.get('Thời điểm', '')).lower() else "dubao"
    bf = row.get('cường độ (cấp BF)', 0)
    if pd.isna(bf) or bf < 6: fname = f"vungthap{status}.png"
    elif bf < 8: fname = f"atnd{status if status=='daqua' else ''}.PNG"
    elif bf <= 11: fname = f"bnd{status if status=='daqua' else ''}.PNG"
    else: fname = f"sieubao{status if status=='daqua' else ''}.PNG"
    path = os.path.join(ICON_DIR, fname)
    return folium.CustomIcon(path, icon_size=(35, 35) if bf>=8 else (22, 22)) if os.path.exists(path) else None

def get_color_by_wind(kt):
    if pd.isna(kt): return 'gray'
    if kt < 34: return '#00CCFF'
    if kt < 64: return '#00FF00'
    if kt < 83: return '#FFFF00'
    if kt < 96: return '#FFAE00'
    if kt < 113: return '#FF0000'
    return '#FF00FF'

# --- 4. HÀM TẠO DASHBOARD (HTML) ---

def create_dashboard_opt1(df, img_b64):
    """Dashboard cho Option 1: Hiện trạng & Dự báo"""
    if df.empty:
        return """
        <div class="dashboard-box" style="position: fixed; top: 20px; right: 20px; width: 300px; background: rgba(255,255,255,0.95); padding: 10px; border-radius: 8px; border: 1px solid #ccc;">
            <div style="text-align:center; color:#d63384; font-weight:bold;">CHƯA CÓ DỮ LIỆU BÃO</div>
            <div style="text-align:center; font-size:12px;">Vui lòng kiểm tra file besttrack.xlsx</div>
        </div>"""

    cur = df[df['Thời điểm'].str.contains("hiện tại", case=False, na=False)]
    fut = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)]
    display_df = pd.concat([cur, fut])

    rows = ""
    for _, r in display_df.iterrows():
        rows += f"""<tr style="background-color: white; border-bottom: 1px solid #ddd;">
<td style="padding:4px; border:1px solid #ccc;">{r.get('Ngày - giờ', '')}</td>
<td style="padding:4px; border:1px solid #ccc;">{r.get('lon', 0):.1f}</td>
<td style="padding:4px; border:1px solid #ccc;">{r.get('lat', 0):.1f}</td>
<td style="padding:4px; border:1px solid #ccc;">{int(r.get('cường độ (cấp BF)', 0))}</td>
<td style="padding:4px; border:1px solid #ccc;">{int(r.get('Pmin (mb)', 0))}</td>
</tr>"""
    
    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:100%; margin-bottom:10px; border-radius:5px;">' if img_b64 else ""

    return textwrap.dedent(f"""
    <div class="dashboard-box" style="position: fixed; top: 20px; right: 20px; width: 320px; background: rgba(255,255,255,0.95); padding: 10px; border-radius: 8px; border: 1px solid #ccc; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
        {img_tag}
        <div style="text-align:center; font-weight:bold; color:#d63384; margin-bottom:5px;">TIN BÃO KHẨN CẤP</div>
        <table style="width:100%; border-collapse: collapse; font-size:11px; text-align:center; color:black;">
            <thead><tr style="background:#007bff; color:white;"><th>Giờ</th><th>Kinh</th><th>Vĩ</th><th>Cấp</th><th>Pmin</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """)

def create_dashboard_opt2(df, selected_storms):
    """Dashboard cho Option 2: Lịch sử"""
    if df.empty or not selected_storms:
        return """
        <div class="dashboard-box" style="position: fixed; top: 20px; right: 20px; width: 250px; background: rgba(255,255,255,0.95); padding: 10px; border-radius: 8px;">
            <div style="background:#007bff; color:white; padding:8px; font-weight:bold;">🌪️ LỊCH SỬ BÃO</div>
            <div style="padding:10px; text-align:center; color:#333;">Chưa chọn bão.</div>
        </div>"""

    rows = ""
    for storm in selected_storms:
        sub = df[df['name'] == storm].sort_values('dt', ascending=False)
        if sub.empty: continue
        latest = sub.iloc[0]
        rows += f"""<tr style="border-bottom:1px solid #eee;">
<td style="padding:5px; color:#007bff; font-weight:bold;">{storm}</td>
<td>{latest['dt'].strftime('%Y-%m-%d')}</td>
<td><span style="background:{get_color_by_wind(latest.get('wind_kt',0))}; padding:2px 5px; border-radius:3px;">{int(latest.get('wind_kt',0))}kt</span></td>
</tr>"""
    
    html = f"""
    <div id="dashboard-opt2" class="dashboard-box" style="position: fixed; top: 20px; right: 20px; width: 300px; background: rgba(255,255,255,0.95); border-radius: 8px; border: 1px solid #ccc; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
        <div style="background:#007bff; color:white; padding:10px; border-radius: 8px 8px 0 0; display:flex; justify-content:space-between; align-items:center; cursor:pointer;" onclick="toggleOpt2()">
            <span style="font-weight:bold;">🌪️ LỊCH SỬ BÃO ({len(selected_storms)})</span>
            <span id="icon-opt2" style="font-size:16px;">➖</span>
        </div>
        <div id="content-opt2" style="padding:10px; max-height:400px; overflow:auto;">
            <table style="width:100%; font-size:12px; text-align:center; border-collapse: collapse; color:black;">
                <tr style="background:#f0f0f0; border-bottom:1px solid #ddd;">
                    <th style="padding:5px;">Tên</th><th>Ngày</th><th>Gió</th>
                </tr>
                {rows}
            </table>
        </div>
    </div>
    <script>
    function toggleOpt2() {{
        var content = document.getElementById('content-opt2');
        var icon = document.getElementById('icon-opt2');
        if (content.style.display === 'none') {{
            content.style.display = 'block';
            icon.innerHTML = '➖';
        }} else {{
            content.style.display = 'none';
            icon.innerHTML = '➕';
        }}
    }}
    </script>
    """
    return textwrap.dedent(html)

# --- 5. CHƯƠNG TRÌNH CHÍNH (MAIN) ---

def main():
    with st.sidebar:
        st.title("🌪️ TRUNG TÂM ĐIỀU KHIỂN")
        st.markdown("---")
        
        # 1. BỘ CHỌN CHẾ ĐỘ (2 OPS CHỌN)
        mode = st.radio(
            "📍 CHỌN CHẾ ĐỘ HIỂN THỊ:",
            ["Option 1: Hiện trạng & Dự báo", "Option 2: Lịch sử & Thống kê"]
        )
        st.markdown("---")
        
        final_df = pd.DataFrame()
        selected_storms = []
        storm_col = None

        # --- XỬ LÝ OPTION 1 ---
        if "Option 1" in mode:
            st.markdown("### 📂 Dữ liệu Option 1")
            f = st.file_uploader("Tải file besttrack.xlsx", type="xlsx", key="opt1")
            file_path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
            
            if file_path:
                try:
                    df = pd.read_excel(file_path)
                    df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(subset=['lat', 'lon'])
                    
                    storm_col = 'Số hiệu' if 'Số hiệu' in df.columns else None
                    if storm_col:
                        st.markdown("### 🌪️ Chọn Bão")
                        all_s = df[storm_col].unique()
                        selected_storms = [s for s in all_s if st.checkbox(f"Bão số {s}", value=True)]
                        final_df = df[df[storm_col].isin(selected_storms)]
                    else: 
                        final_df = df
                        selected_storms = ["Unknown"]
                except Exception as e:
                    st.error(f"Lỗi file: {e}")
            else:
                st.warning("Vui lòng tải file dữ liệu.")

        # --- XỬ LÝ OPTION 2 ---
        else:
            st.markdown("### 📂 Dữ liệu Option 2")
            f = st.file_uploader("Tải file besttrack_capgio.xlsx", type="xlsx", key="opt2")
            file_path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
            
            if file_path:
                try:
                    df = pd.read_excel(file_path)
                    renames = {"tên bão":"name","năm":"year","tháng":"mon","ngày":"day","giờ":"hour","vĩ độ":"lat","kinh độ":"lon","gió (kt)":"wind_kt"}
                    df = df.rename(columns={k:v for k,v in renames.items() if k in df.columns})
                    
                    # Xử lý ngày tháng an toàn
                    time_cols = ['year','mon','day','hour']
                    if all(c in df.columns for c in time_cols):
                        for c in time_cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
                        df['dt'] = pd.to_datetime(df[time_cols].rename(columns={'mon':'month'}))
                    
                    df[['lat','lon','wind_kt']] = df[['lat','lon','wind_kt']].apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(subset=['lat','lon'])
                    
                    # Bộ lọc
                    st.markdown("### 🛠️ Bộ lọc")
                    years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                    temp = df[df['year'].isin(years)]
                    
                    all_storms = temp['name'].unique()
                    selected_storms = st.multiselect("Bão:", all_storms, default=all_storms)
                    final_df = temp[temp['name'].isin(selected_storms)]
                except Exception as e:
                    st.error(f"Lỗi file: {e}")
            else:
                st.warning("Vui lòng tải file dữ liệu.")

    # --- KHỞI TẠO BẢN ĐỒ ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None)
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)

    # --- VẼ LỚP DỮ LIỆU ---
    if not final_df.empty:
        if "Option 1" in mode:
            # === VẼ OPTION 1 ===
            groups = selected_storms if selected_storms else [None]
            for sid in groups:
                sub = final_df[final_df[storm_col] == sid] if storm_col else final_df
                if sub.empty: continue
                
                dense = densify_track(sub)
                f6, f10, fc = create_storm_swaths(dense)
                for geom, col, op in [(f6,COL_R6,0.4), (f10,COL_R10,0.5), (fc,COL_RC,0.6)]:
                    if geom and not geom.is_empty:
                        folium.GeoJson(mapping(geom), style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':0,'fillOpacity':o}).add_to(m)
                
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color="black", weight=2).add_to(m)
                for _, r in sub.iterrows():
                    icon = get_storm_icon(r)
                    if icon: folium.Marker([r['lat'], r['lon']], icon=icon).add_to(m)
            
            # Dashboard Opt 1
            img_b64 = None
            if os.path.exists(CHUTHICH_IMG):
                with open(CHUTHICH_IMG, "rb") as f: img_b64 = base64.b64encode(f.read()).decode()
            st.markdown(create_dashboard_opt1(final_df, img_b64), unsafe_allow_html=True)
            
        else:
            # === VẼ OPTION 2 ===
            for name in selected_storms:
                sub = final_df[final_df['name'] == name].sort_values('dt')
                if sub.empty: continue
                
                coords = sub[['lat','lon']].values.tolist()
                folium.PolyLine(coords, color='black', weight=2, opacity=0.5).add_to(m)
                
                for _, r in sub.iterrows():
                    c = get_color_by_wind(r.get('wind_kt',0))
                    folium.CircleMarker([r['lat'],r['lon']], radius=5, color=c, fill=True, fill_opacity=1, 
                                      popup=f"{name}: {int(r.get('wind_kt',0))}kt").add_to(m)
            
            # Dashboard Opt 2
            st.markdown(create_dashboard_opt2(final_df, selected_storms), unsafe_allow_html=True)
    else:
        # Nếu chưa có dữ liệu, hiển thị Dashboard trống tương ứng
        if "Option 1" in mode:
            st.markdown(create_dashboard_opt1(pd.DataFrame(), None), unsafe_allow_html=True)
        else:
            st.markdown(create_dashboard_opt2(pd.DataFrame(), []), unsafe_allow_html=True)

    # Hiển thị LayerControl và Map
    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, width=None, height=None, use_container_width=True)

if __name__ == "__main__":
    main()
