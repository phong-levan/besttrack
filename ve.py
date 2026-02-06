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

# Thư viện cho Option 1
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH ---
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"        # File Hiện trạng
FILE_OPT2 = "besttrack_capgio.xlsx" # File Lịch sử
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(page_title="Hệ thống Giám sát Bão", layout="wide", initial_sidebar_state="collapsed")

# --- 2. CSS SỬA LỖI HIỂN THỊ (XUYÊN THẤU) ---
st.markdown("""
    <style>
    /* Xóa nền trắng mặc định */
    .stApp, [data-testid="stAppViewContainer"] { background: transparent !important; }
    header, footer { display: none !important; }
    
    /* Reset lề màn hình */
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    
    /* Bản đồ nằm lớp dưới cùng */
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; z-index: 0; }
    
    /* Dashboard & Sidebar nằm lớp trên cùng */
    [data-testid="stSidebar"] { z-index: 10000 !important; background-color: rgba(28, 35, 49, 0.95) !important; }
    .dashboard-box { z-index: 9999 !important; }
    
    /* Fix lỗi hiển thị bảng HTML */
    table { width: 100%; border-collapse: collapse; background: white; }
    td, th { padding: 4px; border: 1px solid #ccc; text-align: center; font-size: 11px; color: black; }
    </style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ SỐ LIỆU ---

def kt_to_bf(kt):
    """Đổi gió (kt) sang cấp Beaufort (BF)"""
    if pd.isna(kt): return 0
    kt = float(kt)
    if kt < 1: return 0
    if kt < 6: return 1
    if kt < 11: return 2
    if kt < 17: return 3
    if kt < 22: return 4
    if kt < 28: return 5
    if kt < 34: return 6
    if kt < 41: return 7
    if kt < 48: return 8
    if kt < 56: return 9
    if kt < 64: return 10
    if kt < 72: return 11 
    return 12             

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

# --- LOGIC ICON BÃO ---
def get_icon_name(row):
    wind_speed = row.get('cuong_do_bf', 0)
    status = row.get('color_key', 'daqua')
    
    if pd.isna(wind_speed): return f"vungthap_{status}"
    if wind_speed < 6:      return f"vungthap_{status}"
    if wind_speed < 8:      return f"atnd_{status}"
    if wind_speed <= 11:    return f"bnd_{status}"
    return f"sieubao_{status}"

# --- 4. HÀM TẠO DASHBOARD (TÁCH RIÊNG & CÁCH XA) ---

def create_dashboard_opt1(df, img_b64):
    """Option 1: Bảng tin (Góc Trên) & Chú thích (Góc Dưới)"""
    
    # --- PHẦN 1: BẢNG TIN BÃO (TOP RIGHT) ---
    table_html = ""
    if df.empty:
        # Hộp cảnh báo nếu chưa có dữ liệu
        table_html = """
        <div class="dashboard-box" style="position: fixed; top: 10px; right: 10px; width: 300px; background: rgba(255,255,255,0.95); padding: 10px; border-radius: 8px; border: 1px solid #ccc;">
            <div style="text-align:center; color:#d63384; font-weight:bold;">CHƯA CÓ DỮ LIỆU BÃO</div>
            <div style="text-align:center; font-size:12px;">Vui lòng tải file besttrack.xlsx</div>
        </div>"""
    else:
        # Lọc dữ liệu hiển thị
        cur = df[df['Thời điểm'].str.contains("hiện tại", case=False, na=False)]
        fut = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)]
        display_df = pd.concat([cur, fut])

        rows = ""
        for _, r in display_df.iterrows():
            rows += f"""<tr style="background-color: white;">
    <td>{r.get('Ngày - giờ', '')}</td>
    <td>{r.get('lon', 0):.1f}</td>
    <td>{r.get('lat', 0):.1f}</td>
    <td>{int(r.get('cường độ (cấp BF)', 0))}</td>
    <td>{int(r.get('Pmin (mb)', 0))}</td>
    </tr>"""
        
        # HTML Bảng tin (Có max-height để không trôi xuống che Chú thích)
        table_html = f"""
        <div class="dashboard-box" style="position: fixed; top: 10px; right: 10px; width: 320px; max-height: 55vh; overflow-y: auto; background: rgba(255,255,255,0.95); padding: 10px; border-radius: 8px; border: 1px solid #ccc; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
            <div style="text-align:center; font-weight:bold; color:#d63384; margin-bottom:5px;">TIN BÃO KHẨN CẤP</div>
            <table>
                <thead><tr style="background:#007bff; color:white;"><th>Giờ</th><th>Kinh</th><th>Vĩ</th><th>Cấp</th><th>Pmin</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    # --- PHẦN 2: CHÚ THÍCH (BOTTOM RIGHT) ---
    legend_html = ""
    if img_b64:
        # Nằm góc dưới cùng bên phải, cách bảng tin một khoảng lớn
        legend_html = f"""
        <div class="dashboard-box" style="position: fixed; bottom: 20px; right: 10px; width: 250px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; border: 1px solid #ccc; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
            <div style="text-align:center; font-weight:bold; font-size:12px; margin-bottom:5px; color:#333;">CHÚ GIẢI</div>
            <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:4px;">
        </div>
        """

    # Trả về cả 2 khối HTML độc lập
    return textwrap.dedent(table_html + legend_html)

def create_dashboard_opt2(df, selected_storms):
    """Option 2: Bảng lịch sử (Góc Trên Phải)"""
    if df.empty or not selected_storms:
        return """
        <div class="dashboard-box" style="position: fixed; top: 10px; right: 10px; width: 250px; background: rgba(255,255,255,0.95); padding: 10px; border-radius: 8px;">
            <div style="background:#007bff; color:white; padding:8px; font-weight:bold;">🌪️ LỊCH SỬ BÃO</div>
            <div style="padding:10px; text-align:center; color:#333;">Chưa chọn bão.</div>
        </div>"""

    rows = ""
    for storm in selected_storms:
        sub = df[df['name'] == storm].sort_values('dt', ascending=False)
        if sub.empty: continue
        latest = sub.iloc[0]
        # Màu nền cho cấp gió
        w = latest.get('wind_kt', 0)
        bg = '#ccc'
        if w >= 64: bg = '#FF00FF'
        elif w >= 48: bg = '#FF0000'
        elif w >= 34: bg = '#FFFF00'
        
        rows += f"""<tr style="border-bottom:1px solid #eee;">
<td style="color:#007bff; font-weight:bold;">{storm}</td>
<td>{latest['dt'].strftime('%Y-%m-%d')}</td>
<td><span style="background:{bg}; padding:2px 5px; border-radius:3px; color:black;">{int(w)}kt</span></td>
</tr>"""
    
    html = f"""
    <div id="dashboard-opt2" class="dashboard-box" style="position: fixed; top: 10px; right: 10px; width: 300px; background: rgba(255,255,255,0.95); border-radius: 8px; border: 1px solid #ccc; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
        <div style="background:#007bff; color:white; padding:10px; border-radius: 8px 8px 0 0; display:flex; justify-content:space-between; align-items:center; cursor:pointer;" onclick="toggleOpt2()">
            <span style="font-weight:bold;">🌪️ LỊCH SỬ BÃO ({len(selected_storms)})</span>
            <span id="icon-opt2" style="font-size:16px;">➖</span>
        </div>
        <div id="content-opt2" style="padding:10px; max-height:60vh; overflow:auto;">
            <table>
                <tr style="background:#f0f0f0;"><th>Tên</th><th>Ngày</th><th>Gió</th></tr>
                {rows}
            </table>
        </div>
    </div>
    <script>
    function toggleOpt2() {{
        var c = document.getElementById('content-opt2');
        var i = document.getElementById('icon-opt2');
        if (c.style.display === 'none') {{ c.style.display = 'block'; i.innerHTML = '➖'; }} 
        else {{ c.style.display = 'none'; i.innerHTML = '➕'; }}
    }}
    </script>
    """
    return textwrap.dedent(html)

# --- 5. MAIN APP ---

def main():
    with st.sidebar:
        st.title("🌪️ TRUNG TÂM ĐIỀU KHIỂN")
        st.markdown("---")
        mode = st.radio("📍 CHỌN CHẾ ĐỘ:", ["Option 1: Hiện trạng & Dự báo", "Option 2: Lịch sử & Thống kê"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        selected_storms = []
        storm_col = None

        # --- XỬ LÝ OPTION 1 ---
        if "Option 1" in mode:
            st.markdown("### 📂 Dữ liệu Hiện trạng")
            f = st.file_uploader("Tải file besttrack.xlsx", type="xlsx")
            file_path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
            
            if file_path:
                df = pd.read_excel(file_path)
                df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
                df = df.dropna(subset=['lat', 'lon'])
                
                if 'cường độ (cấp BF)' in df.columns:
                    df['cuong_do_bf'] = pd.to_numeric(df['cường độ (cấp BF)'], errors='coerce')
                else: df['cuong_do_bf'] = 0
                
                df['color_key'] = df['Thời điểm'].apply(lambda x: 'dubao' if 'dự báo' in str(x).lower() else 'daqua')
                
                storm_col = 'Số hiệu' if 'Số hiệu' in df.columns else None
                if storm_col:
                    all_s = df[storm_col].unique()
                    selected_storms = [s for s in all_s if st.checkbox(f"Bão số {s}", value=True)]
                    final_df = df[df[storm_col].isin(selected_storms)]
                else: final_df = df
            else:
                st.warning("Vui lòng tải file dữ liệu.")

        # --- XỬ LÝ OPTION 2 ---
        else:
            st.markdown("### 📂 Dữ liệu Lịch sử")
            f = st.file_uploader("Tải file besttrack_capgio.xlsx", type="xlsx")
            file_path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
            
            if file_path:
                df = pd.read_excel(file_path)
                renames = {"tên bão":"name","năm":"year","tháng":"mon","ngày":"day","giờ":"hour","vĩ độ":"lat","kinh độ":"lon","gió (kt)":"wind_kt"}
                df = df.rename(columns={k:v for k,v in renames.items() if k in df.columns})
                
                time_cols = ['year','mon','day','hour']
                if all(c in df.columns for c in time_cols):
                    for c in time_cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
                    df['dt'] = pd.to_datetime(df[time_cols].rename(columns={'mon':'month'}))
                
                df[['lat','lon','wind_kt']] = df[['lat','lon','wind_kt']].apply(pd.to_numeric, errors='coerce')
                df['cuong_do_bf'] = df['wind_kt'].apply(kt_to_bf)
                df['color_key'] = 'daqua'
                
                df = df.dropna(subset=['lat','lon'])
                
                years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                temp = df[df['year'].isin(years)]
                
                all_storms = temp['name'].unique()
                selected_storms = st.multiselect("Bão:", all_storms, default=all_storms)
                final_df = temp[temp['name'].isin(selected_storms)]
            else:
                st.warning("Vui lòng tải file dữ liệu.")

    # --- VẼ BẢN ĐỒ ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None)
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)

    fg_icons = folium.FeatureGroup(name="🌀 Biểu tượng Bão")

    if not final_df.empty:
        if "Option 1" in mode:
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

                for _, row in sub.iterrows():
                    icon_name = get_icon_name(row)
                    icon_path = os.path.join(ICON_DIR, f"{icon_name}.png")
                    
                    popup_html = f"""<div style='width:150px'>
                        <b>{row.get('Số hiệu','Bão')}</b><br>
                        Time: {row.get('Ngày - giờ','')}<br>
                        Cấp: {int(row.get('cuong_do_bf',0))}
                    </div>"""
                    
                    if os.path.exists(icon_path):
                        icon = folium.CustomIcon(icon_path, icon_size=(35, 35) if 'sieubao' in icon_name else (25,25))
                        folium.Marker([row['lat'], row['lon']], icon=icon, popup=popup_html).add_to(fg_icons)
                    else:
                        folium.CircleMarker([row['lat'], row['lon']], radius=3, color='black', fill=True, popup=popup_html).add_to(fg_icons)

            # --- DASHBOARD OPTION 1 (Góc trên & dưới phải) ---
            img_b64 = None
            if os.path.exists(CHUTHICH_IMG):
                with open(CHUTHICH_IMG, "rb") as f: img_b64 = base64.b64encode(f.read()).decode()
            st.markdown(create_dashboard_opt1(final_df, img_b64), unsafe_allow_html=True)

        else:
            # OPTION 2
            for name in selected_storms:
                sub = final_df[final_df['name'] == name].sort_values('dt')
                if sub.empty: continue
                
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2, opacity=0.5).add_to(m)
                
                for _, row in sub.iterrows():
                    icon_name = get_icon_name(row)
                    icon_path = os.path.join(ICON_DIR, f"{icon_name}.png")
                    
                    popup_html = f"{name}: {int(row.get('wind_kt',0))}kt"
                    
                    if os.path.exists(icon_path):
                         icon = folium.CustomIcon(icon_path, icon_size=(20, 20))
                         folium.Marker([row['lat'], row['lon']], icon=icon, popup=popup_html).add_to(fg_icons)
                    else:
                         folium.CircleMarker([row['lat'], row['lon']], radius=4, color='red', fill=True, popup=popup_html).add_to(fg_icons)
            
            # --- DASHBOARD OPTION 2 ---
            st.markdown(create_dashboard_opt2(final_df, selected_storms), unsafe_allow_html=True)
            
    else:
        if "Option 1" in mode:
             st.markdown(create_dashboard_opt1(pd.DataFrame(), None), unsafe_allow_html=True)
        else:
             st.markdown(create_dashboard_opt2(pd.DataFrame(), []), unsafe_allow_html=True)

    fg_icons.add_to(m)
    
    # DI CHUYỂN LAYER CONTROL XUỐNG DƯỚI TRÁI ĐỂ TRÁNH CHE KHUẤT
    folium.LayerControl(position='bottomleft', collapsed=True).add_to(m)
    
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
