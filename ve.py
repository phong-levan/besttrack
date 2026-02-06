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

# Thư viện cho Option 1 (Vùng gió)
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH HỆ THỐNG ---
ICON_DIR = "icon"
FILE_OPTION_1 = "besttrack.xlsx"        # File cho hiện trạng/dự báo
FILE_OPTION_2 = "besttrack_capgio.xlsx" # File cho lịch sử
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")

# Màu sắc cho Option 1
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(
    page_title="Hệ thống Theo dõi Bão - Full Screen",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS FULL SCREEN (Áp dụng chung) ---
st.markdown("""
    <style>
    /* Xóa lề, padding */
    html, body, [data-testid="stAppViewContainer"], .block-container {
        padding: 0 !important; margin: 0 !important;
        width: 100vw !important; height: 100vh !important;
        overflow: hidden !important;
    }
    /* Ẩn Header/Footer */
    header, footer { display: none !important; }
    /* Sidebar tối màu */
    [data-testid="stSidebar"] {
        background-color: #1c2331; color: white; z-index: 1000;
    }
    [data-testid="stSidebar"] h1, h2, h3 { color: #00d4ff !important; }
    .stMarkdown, .stText, label, .stRadio label { color: #e0e0e0 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
#  PHẦN XỬ LÝ DỮ LIỆU & HÀM HỖ TRỢ CHUNG
# ==============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def get_color_by_wind(wind_kt):
    """Dùng cho Option 2 (Lịch sử)"""
    if pd.isna(wind_kt): return 'gray'
    w = float(wind_kt)
    if w < 34: return '#00CCFF'
    if w < 64: return '#00FF00'
    if w < 83: return '#FFFF00'
    if w < 96: return '#FFAE00'
    if w < 113: return '#FF0000'
    if w < 137: return '#FF00FF'
    return '#800080'

# ==============================================================================
#  LOGIC OPTION 1: HIỆN TRẠNG & DỰ BÁO (Vùng gió, Icon tùy chỉnh)
# ==============================================================================

def densify_track(df, step_km=10):
    new_rows = []
    if len(df) < 2: return df
    for i in range(len(df) - 1):
        p1, p2 = df.iloc[i], df.iloc[i+1]
        dist = haversine_km(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        n_steps = max(1, int(np.ceil(dist / step_km)))
        for j in range(n_steps):
            f = j / n_steps
            new_rows.append({
                'lat': p1['lat'] + (p2['lat'] - p1['lat']) * f,
                'lon': p1['lon'] + (p2['lon'] - p1['lon']) * f,
                'r6': p1.get('bán kính gió mạnh cấp 6 (km)', 0)*(1-f) + p2.get('bán kính gió mạnh cấp 6 (km)', 0)*f,
                'r10': p1.get('bán kính gió mạnh cấp 10 (km)', 0)*(1-f) + p2.get('bán kính gió mạnh cấp 10 (km)', 0)*f,
                'rc': p1.get('bán kính tâm (km)', 0)*(1-f) + p2.get('bán kính tâm (km)', 0)*f
            })
    new_rows.append(df.iloc[-1].to_dict())
    return pd.DataFrame(new_rows)

def get_storm_icon(row):
    status = "daqua" if "quá khứ" in str(row.get('Thời điểm', '')).lower() else "dubao"
    bf = row.get('cường độ (cấp BF)', 0)
    if pd.isna(bf) or bf < 6: fname = f"vungthap{status}.png"
    elif bf < 8: fname = "atnddaqua.PNG" if status == "daqua" else "atnd.PNG"
    elif bf <= 11: fname = "bnddaqua.PNG" if status == "daqua" else "bnd.PNG"
    else: fname = "sieubaodaqua.PNG" if status == "daqua" else "sieubao.PNG"
    path = os.path.join(ICON_DIR, fname)
    return folium.CustomIcon(path, icon_size=(35, 35) if bf >= 8 else (22, 22)) if os.path.exists(path) else None

def create_storm_swaths(dense_df):
    polys_r6, polys_r10, polys_rc = [], [], []
    geo = geodesic.Geodesic()
    for _, row in dense_df.iterrows():
        for r, target_list in [(row.get('r6', 0), polys_r6), (row.get('r10', 0), polys_r10), (row.get('rc', 0), polys_rc)]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=60)
                target_list.append(Polygon(circle))
    u6 = unary_union(polys_r6) if polys_r6 else None
    u10 = unary_union(polys_r10) if polys_r10 else None
    uc = unary_union(polys_rc) if polys_rc else None
    return (u6.difference(u10) if u6 and u10 else u6), (u10.difference(uc) if u10 and uc else u10), uc

def get_dashboard_opt1(df, img_base64):
    """HTML Dashboard cho Option 1 (Có ảnh chú thích + Bảng tin)"""
    current_df = df[df['Thời điểm'].str.contains("hiện tại", case=False, na=False)]
    forecast_df = df[df['Thời điểm'].str.contains("dự báo", case=False, na=False)]
    display_df = pd.concat([current_df, forecast_df])
    
    rows_html = "".join([f"""
        <tr style="background: #ffffff;">
            <td style="border: 1px solid #333; padding: 4px;">{r['Ngày - giờ']}</td>
            <td style="border: 1px solid #333; padding: 4px;">{float(r['lon']):.1f}E</td>
            <td style="border: 1px solid #333; padding: 4px;">{float(r['lat']):.1f}N</td>
            <td style="border: 1px solid #333; padding: 4px;">Cấp {int(r['cường độ (cấp BF)'])}</td>
            <td style="border: 1px solid #333; padding: 4px;">{int(r.get('Pmin (mb)', 0))}</td>
        </tr>""" for _, r in display_df.iterrows()])
    
    img_html = f'<img src="data:image/png;base64,{img_base64}" style="width: 100%; border-radius: 5px; margin-bottom: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">' if img_base64 else ""
    
    return f"""
    <div style="position: fixed; top: 15px; right: 15px; width: 320px; z-index: 9999;">
        {img_html}
        <div style="background: rgba(255,255,255,0.95); border: 2px solid #333; border-radius: 5px; padding: 8px; font-family: Arial;">
            <div style="text-align: center; font-size: 13px; font-weight: bold; margin-bottom: 5px; color: black;">TIN BÃO KHẨN CẤP</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 11px; text-align: center; color: black;">
                <tr style="background: #e0e0e0; border: 1px solid #333;">
                    <th>Giờ</th><th>Kinh</th><th>Vĩ</th><th>Cấp</th><th>Pmin</th>
                </tr>
                {rows_html}
            </table>
        </div>
    </div>"""

# ==============================================================================
#  LOGIC OPTION 2: LỊCH SỬ BÃO (Đường màu, Dashboard co giãn)
# ==============================================================================

def get_dashboard_opt2(df, selected_storms):
    """HTML Dashboard cho Option 2 (Co giãn, Scroll)"""
    if df.empty: return ""
    box_id, content_id = "box-opt2", "content-opt2"
    
    rows = ""
    for storm_name in selected_storms:
        sub = df[df['name'] == storm_name].sort_values('dt', ascending=False)
        if sub.empty: continue
        latest = sub.iloc[0]
        rows += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="color:#007bff; font-weight:bold;">{storm_name}</td>
            <td>{latest['dt'].strftime('%d/%m/%Y')}</td>
            <td><span style="background:{get_color_by_wind(latest.get('wind_kt',0))}; padding:2px 5px; border-radius:3px; font-size:10px;">{int(latest.get('wind_kt',0))}kt</span></td>
        </tr>"""
        
    return f"""
    <div id="{box_id}" style="position: fixed; top: 20px; right: 20px; width: 280px; z-index: 9999; 
        background: rgba(255,255,255,0.95); border-radius: 8px; border: 1px solid #ccc; font-family: Arial; resize: both; overflow: auto;">
        <div style="background: #007bff; color: white; padding: 8px; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; cursor: pointer;" onclick="document.getElementById('{content_id}').style.display = document.getElementById('{content_id}').style.display==='none'?'block':'none'">
            <span style="font-weight:bold;">THỐNG KÊ LỊCH SỬ</span><span>▼</span>
        </div>
        <div id="{content_id}" style="padding: 10px;">
            <table style="width: 100%; font-size: 12px; color:#333; text-align:center;">
                <thead><tr style="background:#f0f0f0;"><th>Tên</th><th>Ngày</th><th>Gió</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>"""

# ==============================================================================
#  MAIN APP
# ==============================================================================

def main():
    # --- SIDEBAR: CHỌN OPTION ---
    with st.sidebar:
        st.title("🌪️ CONTROL PANEL")
        
        # 1. Chọn chế độ hiển thị
        mode = st.radio(
            "Chọn chế độ xem:",
            ("Option 1: Hiện trạng & Dự báo", "Option 2: Lịch sử & Thống kê")
        )
        st.markdown("---")

        # 2. Xử lý dữ liệu dựa trên Mode
        if "Option 1" in mode:
            # --- LOGIC OPTION 1 ---
            data_file = FILE_OPTION_1
            uploaded = st.file_uploader("Upload file Hiện trạng (xlsx)", type=["xlsx"])
            if uploaded: data_file = uploaded
            
            if os.path.exists(str(data_file)) or uploaded:
                raw_df = pd.read_excel(data_file)
                # Xử lý tọa độ
                raw_df[['lat', 'lon']] = raw_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
                raw_df = raw_df.dropna(subset=['lat', 'lon'])
                
                # Chọn bão
                storm_col = 'Số hiệu' if 'Số hiệu' in raw_df.columns else None
                selected_storms = []
                if storm_col:
                    st.write("<b>Chọn bão hiển thị:</b>", unsafe_allow_html=True)
                    for s in raw_df[storm_col].unique():
                        if st.checkbox(f"Bão số {s}", value=True): selected_storms.append(s)
                    final_df = raw_df[raw_df[storm_col].isin(selected_storms)]
                else:
                    final_df = raw_df
            else:
                st.warning(f"Chưa có file {FILE_OPTION_1}"); final_df = pd.DataFrame()

        else:
            # --- LOGIC OPTION 2 ---
            data_file = FILE_OPTION_2
            uploaded = st.file_uploader("Upload file Lịch sử (xlsx)", type=["xlsx"])
            if uploaded: data_file = uploaded
            
            if os.path.exists(str(data_file)) or uploaded:
                df_hist = pd.read_excel(data_file)
                # Rename & Process
                rename_map = {"tên bão":"name", "năm":"year", "tháng":"mon", "ngày":"day", "giờ":"hour", 
                              "vĩ độ":"lat", "kinh độ":"lon", "gió (kt)":"wind_kt"}
                df_hist = df_hist.rename(columns={k:v for k,v in rename_map.items() if k in df_hist.columns})
                
                # Xử lý thời gian
                if 'year' in df_hist.columns:
                    df_hist['dt'] = pd.to_datetime(df_hist[['year','mon','day','hour']].astype(str).agg('-'.join, axis=1)+':00', format='%Y-%m-%d-%H:%00', errors='coerce')
                
                # Filter Sidebar
                years = st.multiselect("Năm:", sorted(df_hist['year'].unique()), default=sorted(df_hist['year'].unique())[-1:])
                temp_df = df_hist[df_hist['year'].isin(years)]
                selected_storms = st.multiselect("Bão:", temp_df['name'].unique(), default=temp_df['name'].unique())
                final_df = temp_df[temp_df['name'].isin(selected_storms)]
            else:
                st.warning(f"Chưa có file {FILE_OPTION_2}"); final_df = pd.DataFrame()

    # --- MAIN MAP: HIỂN THỊ DỰA TRÊN OPTION ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None)
    
    # Layer Control cơ bản
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)

    if not final_df.empty:
        if "Option 1" in mode:
            # === VẼ CHO OPTION 1 (Polygon gió, Icon Custom) ===
            groups = selected_storms if 'selected_storms' in locals() and selected_storms else [None]
            
            for storm_id in groups:
                # Lọc dữ liệu từng cơn bão
                storm_data = final_df[final_df[storm_col] == storm_id] if storm_col else final_df
                if storm_data.empty: continue
                
                # 1. Vẽ Vùng gió (Polygon)
                dense_df = densify_track(storm_data)
                f6, f10, fc = create_storm_swaths(dense_df)
                
                fg_wind = folium.FeatureGroup(name=f"Vùng gió Bão {storm_id}")
                for geom, col, op in [(f6, COL_R6, 0.4), (f10, COL_R10, 0.5), (fc, COL_RC, 0.6)]:
                    if geom and not geom.is_empty:
                        folium.GeoJson(mapping(geom), style_function=lambda x,c=col,o=op: {'fillColor':c,'color':c,'weight':1,'fillOpacity':o}).add_to(fg_wind)
                fg_wind.add_to(m)

                # 2. Vẽ Đường đi & Icon
                folium.PolyLine(storm_data[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
                for _, row in storm_data.iterrows():
                    icon = get_storm_icon(row)
                    if icon: folium.Marker([row['lat'], row['lon']], icon=icon).add_to(m)
            
            # 3. Inject Dashboard HTML (Option 1)
            img_b64 = None
            if os.path.exists(CHUTHICH_IMG):
                with open(CHUTHICH_IMG, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
            st.markdown(get_dashboard_opt1(final_df, img_b64), unsafe_allow_html=True)
            
        else:
            # === VẼ CHO OPTION 2 (Đường màu, Dot tròn) ===
            for storm_name in selected_storms:
                sub = final_df[final_df['name'] == storm_name].sort_values('dt')
                if sub.empty: continue
                
                # Vẽ đường
                coords = sub[['lat', 'lon']].values.tolist()
                folium.PolyLine(coords, color='black', weight=2, opacity=0.5).add_to(m)
                
                # Vẽ điểm màu
                for _, row in sub.iterrows():
                    c = get_color_by_wind(row.get('wind_kt', 0))
                    folium.CircleMarker([row['lat'], row['lon']], radius=5, color=c, fill=True, fill_opacity=1, 
                                      popup=f"{storm_name}: {int(row.get('wind_kt',0))}kt").add_to(m)
            
            # 3. Inject Dashboard HTML (Option 2)
            st.markdown(get_dashboard_opt2(final_df, selected_storms), unsafe_allow_html=True)

    # Thêm Layer Control
    folium.LayerControl(collapsed=True).add_to(m)
    
    # Hiển thị Full Screen
    st_folium(m, width=None, height=None, use_container_width=True)

if __name__ == "__main__":
    main()
