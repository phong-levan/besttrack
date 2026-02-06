# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import io
import base64
from math import radians, sin, cos, asin, sqrt
import warnings
import textwrap

# Thư viện xử lý hình học & bản đồ tĩnh
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patheffects as path_effects

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & GIAO DIỆN (INFRASTRUCTURE CONFIG)
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.xlsx"        # Dữ liệu Bão Hiện trạng
FILE_OPT2 = "besttrack_capgio.xlsx" # Dữ liệu Bão Lịch sử
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90"

st.set_page_config(
    page_title="Hệ thống Tích hợp Bão & Thời tiết",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS "Xuyên thấu" & Quy hoạch vị trí 4 góc
st.markdown("""
    <style>
    /* Xóa nền trắng mặc định */
    .stApp, [data-testid="stAppViewContainer"] { background: transparent !important; }
    header, footer { display: none !important; }
    .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
    
    /* Bản đồ nằm lớp dưới cùng */
    iframe { position: fixed; top: 0; left: 0; width: 100vw !important; height: 100vh !important; z-index: 0; }
    
    /* Sidebar nằm lớp trên cùng */
    [data-testid="stSidebar"] { z-index: 10000 !important; background-color: rgba(28, 35, 49, 0.95) !important; }
    
    /* Style cho các hộp thông tin nổi (Dashboard Boxes) */
    .info-box { z-index: 9999 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Tinh chỉnh bảng dữ liệu bên trong Dashboard */
    table { width: 100%; border-collapse: collapse; background: white; font-size: 11px; }
    th { background-color: #007bff; color: white; padding: 5px; border: 1px solid #ccc; }
    td { padding: 4px; border: 1px solid #ccc; text-align: center; color: #333; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CÁC MODULE XỬ LÝ DỮ LIỆU (DATA PROCESSING MODULES)
# ==============================================================================

@st.cache_data
def load_data_storm(file_path):
    """Đọc dữ liệu bão chuẩn hóa"""
    if not os.path.exists(file_path): return None
    df = pd.read_excel(file_path)
    
    # Mapping cột tiếng Việt -> tiếng Anh
    rename_map = {
        "tên bão": "name", "biển đông": "storm_no", "năm": "year", "tháng": "mon", 
        "ngày": "day", "giờ": "hour", "vĩ độ": "lat", "kinh độ": "lon", 
        "gió (kt)": "wind_kt", "khí áp (mb)": "pressure", 
        "Thời điểm": "status_raw", "Ngày - giờ": "datetime_str",
        "bán kính gió mạnh cấp 6 (km)": "r6", "bán kính gió mạnh cấp 10 (km)": "r10", "bán kính tâm (km)": "rc",
        "cường độ (cấp BF)": "bf"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    # Xử lý ngày tháng
    if 'datetime_str' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    elif all(c in df.columns for c in ['year', 'mon', 'day', 'hour']):
        cols = ['year', 'mon', 'day', 'hour']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
        df['dt'] = pd.to_datetime(df[cols].rename(columns={'mon':'month'}))
        
    # Ép kiểu số
    num_cols = ['lat', 'lon', 'wind_kt', 'r6', 'r10', 'rc', 'bf']
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df.dropna(subset=['lat', 'lon'])

# --- Module Tính toán Hình học (Vùng gió) ---
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
            for col in ['r6', 'r10', 'rc', 'bf', 'wind_kt']: # Nội suy các chỉ số
                if col in p1 and col in p2: row[col] = p1.get(col, 0)*(1-f) + p2.get(col, 0)*f
            new_rows.append(row)
    new_rows.append(df.iloc[-1])
    return pd.DataFrame(new_rows)

def create_storm_swaths(dense_df):
    polys = {'r6': [], 'r10': [], 'rc': []}
    geo = geodesic.Geodesic()
    for _, row in dense_df.iterrows():
        for r, key in [(row.get('r6',0), 'r6'), (row.get('r10',0), 'r10'), (row.get('rc',0), 'rc')]:
            if r > 0:
                circle = geo.circle(lon=row['lon'], lat=row['lat'], radius=r*1000, n_samples=30)
                polys[key].append(Polygon(circle))
    u = {k: unary_union(v) if v else None for k, v in polys.items()}
    f_rc = u['rc']
    f_r10 = u['r10'].difference(u['rc']) if u['r10'] and u['rc'] else u['r10']
    f_r6 = u['r6'].difference(u['r10']) if u['r6'] and u['r10'] else u['r6']
    return f_r6, f_r10, f_rc

# --- Module Logic Icon & Màu sắc ---
def get_icon_name(row):
    """Xác định tên icon dựa trên cấp gió BF và trạng thái"""
    # Nếu có cột BF thì dùng, không thì quy đổi từ wind_kt
    bf = row.get('bf', 0)
    if pd.isna(bf) or bf == 0:
        w = row.get('wind_kt', 0)
        if w < 34: bf = 6
        elif w < 64: bf = 8
        else: bf = 12
        
    status = 'dubao' if 'forecast' in str(row.get('status_raw', '')) else 'daqua'
    
    if bf < 6: return f"vungthap_{status}"
    if bf < 8: return f"atnd_{status}"
    if bf <= 11: return f"bnd_{status}"
    return f"sieubao_{status}"

# --- Module Xuất Bản đồ Tĩnh (Matplotlib/Cartopy) ---
def generate_static_map(df, title="SƠ ĐỒ BÃO"):
    fig = plt.figure(figsize=(14, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([98, 125, 5, 25], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, linestyle="--", edgecolor='gray')
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    unique_storms = df['name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_storms), 1)))
    
    for i, storm_name in enumerate(unique_storms):
        sub = df[df['name'] == storm_name].sort_values('dt')
        if sub.empty: continue
        color = colors[i]
        
        ax.plot(sub['lon'], sub['lat'], color=color, linewidth=2, transform=ccrs.PlateCarree(), label=storm_name, zorder=5)
        ax.scatter(sub['lon'], sub['lat'], color=color, s=20, zorder=6, transform=ccrs.PlateCarree())
        
        # Nhãn tên bão
        start_pt = sub.iloc[0]
        ax.text(start_pt['lon'], start_pt['lat'], storm_name, transform=ccrs.PlateCarree(), 
                fontsize=9, color='blue', weight='bold', 
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.legend(loc='upper right', title="Danh sách bão")
    ax.set_title(title, fontsize=15, weight='bold', color='#003366', pad=15)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0); plt.close(fig)
    return buf

# ==============================================================================
# 3. CÁC HÀM TẠO GIAO DIỆN NỔI (DASHBOARD WIDGETS)
# ==============================================================================

def create_info_table_html(df, title="THÔNG TIN CHI TIẾT"):
    """Tạo bảng thông tin ở GÓC TRÊN PHẢI"""
    content = ""
    if df.empty:
        content = "<div style='text-align:center; padding:10px; color:#666;'>Không có dữ liệu hiển thị.</div>"
    else:
        # Logic hiển thị: Nếu là hiện trạng lấy hiện tại+dự báo, nếu lịch sử lấy dòng cuối
        if 'status_raw' in df.columns: # Option 1
            cur = df[df['status_raw'].astype(str).str.contains("hiện tại|current", case=False, na=False)]
            fut = df[df['status_raw'].astype(str).str.contains("dự báo|forecast", case=False, na=False)]
            display_df = pd.concat([cur, fut]).head(10) # Giới hạn 10 dòng
        else: # Option 2
            display_df = df.sort_values('dt', ascending=False).groupby('name').head(1)

        rows = ""
        for _, r in display_df.iterrows():
            t = r.get('datetime_str') if pd.notna(r.get('datetime_str')) else r['dt'].strftime('%d/%m %Hh')
            wind = int(r.get('bf')) if pd.notna(r.get('bf')) and r.get('bf')!=0 else int(r.get('wind_kt', 0))
            rows += f"<tr><td>{t}</td><td>{r['lat']:.1f}/{r['lon']:.1f}</td><td>{wind}</td></tr>"
            
        content = f"""
        <table>
            <thead><tr><th>Thời gian</th><th>Vị trí (N/E)</th><th>Cấp/Gió</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; top: 20px; right: 20px; width: 300px; max-height: 50vh; overflow-y: auto; background: white; border-radius: 8px; border: 1px solid #999; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="background:#007bff; color:white; padding:8px; text-align:center; font-weight:bold;">{title}</div>
        {content}
    </div>
    """)

def create_legend_html(img_b64):
    """Tạo bảng chú thích ở GÓC DƯỚI PHẢI"""
    if not img_b64: return ""
    return textwrap.dedent(f"""
    <div class="info-box" style="position: fixed; bottom: 30px; right: 20px; width: 260px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; border: 1px solid #999; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="text-align:center; font-weight:bold; font-size:12px; margin-bottom:5px; color:#333;">CHÚ GIẢI KÝ HIỆU</div>
        <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:4px;">
    </div>
    """)

# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN APP LOGIC)
# ==============================================================================
def main():
    # --- KHỞI TẠO BẢN ĐỒ ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    fg_storm = folium.FeatureGroup(name="Lớp Bão")
    fg_weather = folium.FeatureGroup(name="Lớp Thời Tiết") # Placeholder cho tương lai

    # --- SIDEBAR: CẤU TRÚC PHÂN CẤP (HIERARCHY) ---
    with st.sidebar:
        st.title("🎛️ MENU ĐIỀU KHIỂN")
        
        # CẤP 1: CHỌN CHỦ ĐỀ
        main_topic = st.selectbox("1. CHỌN CHỦ ĐỀ:", ["Bão (Typhoon)", "Thời tiết (Weather)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        active_mode = ""

        # === NHÁNH 1: BÃO ===
        if main_topic == "Bão (Typhoon)":
            # CẤP 2: CHỨC NĂNG BÃO
            storm_mode = st.radio("2. CHỨC NĂNG:", ["Option 1: Hiện trạng", "Option 2: Lịch sử"])
            active_mode = storm_mode
            
            # --- Option 1: Hiện trạng ---
            if "Option 1" in storm_mode:
                st.info("📂 Đang dùng: besttrack.xlsx")
                f = st.file_uploader("Upload File:", type="xlsx", key="opt1")
                path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                
                show_layer = st.checkbox("Hiển thị lớp Hiện trạng", value=True)
                
                if path:
                    df = load_data_storm(path)
                    if df is not None and not df.empty:
                        # Logic lọc bão
                        if 'storm_no' in df.columns:
                            selected = st.multiselect("Chọn bão:", df['storm_no'].unique(), default=df['storm_no'].unique())
                            final_df = df[df['storm_no'].isin(selected)]
                        else: final_df = df
                        
                        # Logic Vẽ (Layer)
                        if show_layer and not final_df.empty:
                            groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
                            for g in groups:
                                sub = final_df[final_df['storm_no'] == g] if g else final_df
                                if sub.empty: continue
                                
                                # Vẽ vùng gió
                                dense = densify_track(sub)
                                f6, f10, fc = create_storm_swaths(dense)
                                for geom, c, o in [(f6,COL_R6,0.4), (f10,COL_R10,0.5), (fc,COL_RC,0.6)]:
                                    if geom and not geom.is_empty:
                                        folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':0,'fillOpacity':o}).add_to(fg_storm)
                                
                                # Vẽ đường đi
                                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='black', weight=2).add_to(fg_storm)
                                
                                # Vẽ Icon
                                for _, r in sub.iterrows():
                                    icon_name = get_icon_name(r)
                                    icon_path = os.path.join(ICON_DIR, f"{icon_name}.png")
                                    popup = f"Bão số {r.get('storm_no','')}"
                                    if os.path.exists(icon_path):
                                        icon = folium.CustomIcon(icon_path, icon_size=(30, 30))
                                        folium.Marker([r['lat'], r['lon']], icon=icon, popup=popup).add_to(fg_storm)
                                    else:
                                        folium.CircleMarker([r['lat'], r['lon']], radius=3, color='black', fill=True).add_to(fg_storm)
                else:
                    st.warning("Vui lòng tải file besttrack.xlsx")

            # --- Option 2: Lịch sử ---
            else:
                st.info("📂 Đang dùng: besttrack_capgio.xlsx")
                f = st.file_uploader("Upload File:", type="xlsx", key="opt2")
                path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
                
                show_layer = st.checkbox("Hiển thị lớp Lịch sử", value=True)
                
                if path:
                    df = load_data_storm(path)
                    if df is not None and not df.empty:
                        # Logic Lọc
                        years = st.multiselect("Lọc Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                        temp = df[df['year'].isin(years)]
                        names = st.multiselect("Lọc Tên Bão:", temp['name'].unique(), default=temp['name'].unique())
                        final_df = temp[temp['name'].isin(names)]
                        
                        # Logic Vẽ
                        if show_layer and not final_df.empty:
                            for name in final_df['name'].unique():
                                sub = final_df[final_df['name'] == name].sort_values('dt')
                                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='blue', weight=2, opacity=0.6).add_to(fg_storm)
                                for _, r in sub.iterrows():
                                    w = r.get('wind_kt', 0)
                                    c = '#00CCFF' if w<34 else ('#00FF00' if w<64 else ('#FFFF00' if w<83 else '#FF0000'))
                                    folium.CircleMarker([r['lat'],r['lon']], radius=4, color=c, fill=True, fill_opacity=1, popup=f"{name}: {int(w)}kt").add_to(fg_storm)
                else:
                    st.warning("Vui lòng tải file besttrack_capgio.xlsx")
            
            # Export (Chung cho nhánh Bão)
            if not final_df.empty:
                st.markdown("---")
                if st.button("🖼️ Tải ảnh bản đồ (PNG)"):
                    img = generate_static_map(final_df, title=f"BẢN ĐỒ BÃO - {storm_mode}")
                    st.download_button("⬇️ Download PNG", img, "storm_map.png", "image/png")

        # === NHÁNH 2: THỜI TIẾT (KHUNG SƯỜN MỞ RỘNG) ===
        else:
            active_mode = "Thời tiết"
            # CẤP 2: LOẠI DỮ LIỆU
            weather_mode = st.radio("2. LOẠI DỮ LIỆU:", ["Option 3: Quan trắc", "Option 4: Mô hình"])
            
            # CẤP 3: THÔNG SỐ (Chung cho cả Quan trắc/Mô hình)
            param = st.selectbox("3. CHỌN THÔNG SỐ:", [
                "Nhiệt độ (Temp)", 
                "Lượng mưa (Rain)", 
                "Gió (Wind)"
            ])
            st.markdown("---")
            
            # Logic Placeholder (Để bạn điền code xử lý file NetCDF/GRIB sau này)
            st.info(f"Đang chọn: {weather_mode} > {param}")
            if st.checkbox("Hiển thị lớp dữ liệu", value=True):
                # Demo: Vẽ một hình chữ nhật giả lập vùng dữ liệu
                folium.Rectangle(bounds=[[10, 105], [20, 115]], color="orange", fill=True, fill_opacity=0.2, popup=f"Vùng dữ liệu {param}").add_to(fg_weather)

    # --- RENDER GIAO DIỆN TRÊN BẢN ĐỒ ---
    
    # 1. Dashboard (Bảng tin) - Góc Trên Phải
    if not final_df.empty:
        title = "TIN BÃO KHẨN CẤP" if "Option 1" in active_mode else "THỐNG KÊ LỊCH SỬ"
        st.markdown(create_info_table_html(final_df, title), unsafe_allow_html=True)
    elif "Bão" in main_topic:
        st.markdown(create_info_table_html(pd.DataFrame(), "ĐANG CHỜ DỮ LIỆU..."), unsafe_allow_html=True)

    # 2. Legend (Chú thích ảnh) - Góc Dưới Phải (Chỉ hiện cho Option 1 Bão)
    if "Option 1" in active_mode and os.path.exists(CHUTHICH_IMG):
        with open(CHUTHICH_IMG, "rb") as f: img_b64 = base64.b64encode(f.read()).decode()
        st.markdown(create_legend_html(img_b64), unsafe_allow_html=True)

    # 3. Add Layers to Map
    fg_storm.add_to(m)
    fg_weather.add_to(m)
    
    # 4. Layer Control - Góc Dưới Trái
    folium.LayerControl(position='bottomleft', collapsed=True).add_to(m)
    
    # 5. Hiển thị Map
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
