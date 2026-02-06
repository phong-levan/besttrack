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

st.set_page_config(page_title="Hệ thống Giám sát Bão", layout="wide", initial_sidebar_state="expanded")

# --- 2. CSS QUY HOẠCH GIAO DIỆN (TẠO HỘP CÔNG CỤ TRÁI) ---
st.markdown("""
    <style>
    /* Xóa nền trắng mặc định */
    .stApp, [data-testid="stAppViewContainer"] { background: transparent !important; }
    header, footer { display: none !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; z-index: 0; }
    [data-testid="stSidebar"] { z-index: 10000 !important; background-color: rgba(28, 35, 49, 0.95) !important; }
    
    /* --- CSS CHO CÁC HỘP THÔNG TIN (INFO BOXES) --- */
    .info-box { z-index: 9999 !important; }
    
    /* --- BIẾN ĐỔI LAYER CONTROL THÀNH "HỘP CÔNG CỤ" (TOP-LEFT) --- */
    /* 1. Style khung hộp */
    .leaflet-top.leaflet-left .leaflet-control-layers {
        background: rgba(255,255,255,0.95) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
        padding: 10px !important;
        border: 1px solid #999 !important;
        min-width: 180px;
    }
    
    /* 2. Thêm Tiêu đề "HỘP CÔNG CỤ" vào đầu Layer Control */
    .leaflet-control-layers-expanded::before {
        content: "🛠️ HỘP CÔNG CỤ";
        display: block;
        font-weight: bold;
        text-align: center;
        color: #d63384;
        margin-bottom: 8px;
        font-family: Arial, sans-serif;
        font-size: 13px;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }
    
    /* 3. Chỉnh font chữ trong hộp */
    .leaflet-control-layers label {
        font-size: 12px !important;
        font-family: Arial, sans-serif !important;
        color: #333 !important;
    }
    
    /* Style cho bảng dữ liệu */
    table { width: 100%; border-collapse: collapse; background: white; }
    td, th { padding: 5px; border: 1px solid #ddd; text-align: center; font-size: 11px; color: black; }
    th { background-color: #007bff; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ SỐ LIỆU ---

def kt_to_bf(kt):
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

# --- 4. HÀM TẠO HTML DASHBOARD (CÁC GÓC CÒN LẠI) ---

def create_info_table_html(df, title="TIN BÃO KHẨN CẤP"):
    """Tạo bảng thông tin ở GÓC TRÊN PHẢI"""
    if df.empty:
        content = "<div style='text-align:center; padding:10px;'>Chưa có dữ liệu.</div>"
    else:
        if 'Thời điểm' in df.columns:
            cur = df[df['Thời điểm'].str.contains("hiện tại", case=False, na=False)]
            fut = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)]
            display_df = pd.concat([cur, fut])
        else:
            display_df = df.sort_values('dt', ascending=False).groupby('name').head(1)

        rows = ""
        for _, r in display_df.iterrows():
            time_str = r.get('Ngày - giờ') if 'Ngày - giờ' in r else r['dt'].strftime('%d/%m %Hh')
            wind = int(r.get('cường độ (cấp BF)')) if 'cường độ (cấp BF)' in r else int(r.get('wind_kt', 0))
            rows += f"""<tr>
                <td>{time_str}</td>
                <td>{r.get('lon', 0):.1f}</td>
                <td>{r.get('lat', 0):.1f}</td>
                <td>{wind}</td>
            </tr>"""
            
        content = f"""
        <table>
            <thead><tr><th>Thời gian</th><th>Kinh</th><th>Vĩ</th><th>Cấp</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; top: 20px; right: 20px; width: 300px; max-height: 50vh; overflow-y: auto; background: white; border-radius: 8px; border: 1px solid #999; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="background:#007bff; color:white; padding:8px; text-align:center; font-weight:bold; font-family:Arial;">{title}</div>
        {content}
    </div>
    """)

def create_legend_html(img_b64):
    """Tạo bảng chú thích ở GÓC DƯỚI PHẢI"""
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; bottom: 30px; right: 20px; width: 260px; background: white; padding: 10px; border-radius: 8px; border: 1px solid #999; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="text-align:center; font-weight:bold; font-size:12px; margin-bottom:5px; color:#333;">CHÚ GIẢI KÝ HIỆU</div>
        <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:4px;">
    </div>
    """)

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

        if "Option 1" in mode:
            st.info("Đang xem: HIỆN TRẠNG")
            f = st.file_uploader("Tải besttrack.xlsx", type="xlsx")
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

        else:
            st.info("Đang xem: LỊCH SỬ")
            f = st.file_uploader("Tải besttrack_capgio.xlsx", type="xlsx")
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

    # --- KHỞI TẠO BẢN ĐỒ (QUAN TRỌNG: Tắt zoom mặc định để tự thêm ở vị trí khác nếu cần) ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
    
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
                    popup_html = f"<b>{row.get('Số hiệu','Bão')}</b>: Cấp {int(row.get('cuong_do_bf',0))}"
                    
                    if os.path.exists(icon_path):
                        icon = folium.CustomIcon(icon_path, icon_size=(35, 35) if 'sieubao' in icon_name else (25,25))
                        folium.Marker([row['lat'], row['lon']], icon=icon, popup=popup_html).add_to(fg_icons)
                    else:
                        folium.CircleMarker([row['lat'], row['lon']], radius=3, color='black', fill=True, popup=popup_html).add_to(fg_icons)

            # RENDER OPTION 1 DASHBOARD
            st.markdown(create_info_table_html(final_df, "TIN BÃO KHẨN CẤP"), unsafe_allow_html=True)
            if os.path.exists(CHUTHICH_IMG):
                with open(CHUTHICH_IMG, "rb") as f: img_b64 = base64.b64encode(f.read()).decode()
                st.markdown(create_legend_html(img_b64), unsafe_allow_html=True)

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
            
            # RENDER OPTION 2 DASHBOARD
            st.markdown(create_info_table_html(final_df, "THỐNG KÊ LỊCH SỬ"), unsafe_allow_html=True)
            
    else:
        st.markdown(create_info_table_html(pd.DataFrame(), "ĐANG CHỜ DỮ LIỆU..."), unsafe_allow_html=True)

    fg_icons.add_to(m)
    
    # --- ĐỊNH VỊ LAYER CONTROL (HỘP CÔNG CỤ) TẠI TOP-LEFT ---
    # Thuộc tính collapsed=False giúp nó luôn mở ra giống một bảng điều khiển
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)
    
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
