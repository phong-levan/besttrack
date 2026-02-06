# -*- coding: utf-8 -*-
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
import warnings

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(
    page_title="Hệ thống Giám sát Bão",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ICON_DIR = "icon"  # Thư mục chứa icon

# --- 2. CSS TỐI GIẢN ---
st.markdown("""
    <style>
    .block-container { padding: 0 !important; max-width: 100% !important; }
    header {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    [data-testid="stSidebar"] { background-color: #1c2331; color: white; opacity: 0.9; }
    [data-testid="stSidebar"] h1, h2, h3 { color: #00d4ff !important; }
    .stMarkdown, .stText, label { color: #e0e0e0 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---

def kt_to_bf(kt):
    """Chuyển đổi tốc độ gió (kt) sang cấp Beaufort (ước lượng)"""
    if pd.isna(kt): return 0
    if kt < 1: return 0
    if kt < 4: return 1
    if kt < 7: return 2
    if kt < 11: return 3
    if kt < 17: return 4
    if kt < 22: return 5  # < 6 (Vùng thấp)
    if kt < 28: return 6
    if kt < 34: return 7  # < 8 (ATNĐ)
    if kt < 41: return 8
    if kt < 48: return 9
    if kt < 56: return 10
    if kt < 64: return 11 # <= 11 (Bão)
    return 12             # > 11 (Siêu bão)

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_excel(file_path)
    
    rename_map = {
        "tên bão": "name", "biển đông": "storm_no", 
        "năm": "year", "tháng": "mon", "ngày": "day", "giờ": "hour", 
        "vĩ độ": "lat", "kinh độ": "lon", 
        "gió (kt)": "wind_kt", "khí áp (mb)": "pressure", 
        "Thời điểm": "status_raw", "Ngày - giờ": "datetime_str"
    }
    valid_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_rename)
    
    # Xử lý thời gian
    if 'datetime_str' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    elif all(c in df.columns for c in ['year', 'mon', 'day', 'hour']):
        try:
            time_cols = ['year', 'mon', 'day', 'hour']
            for col in time_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=time_cols)
            df[time_cols] = df[time_cols].astype(int)
            temp_df = df[time_cols].rename(columns={'mon': 'month'})
            df['dt'] = pd.to_datetime(temp_df)
        except: pass
            
    # Phân loại trạng thái (status)
    if 'status_raw' in df.columns:
        def categorize(val):
            val_str = str(val).lower()
            if 'dự báo' in val_str: return 'forecast'
            return 'past' # Bao gồm cả hiện tại và quá khứ
        df['status'] = df['status_raw'].apply(categorize)
    else:
        df['status'] = 'past'

    # Tạo cột color_key (daqua / dubao)
    df['color_key'] = df['status'].apply(lambda x: 'dubao' if x == 'forecast' else 'daqua')

    # Xử lý số liệu gió & tính cấp BF
    if 'wind_kt' in df.columns:
        df['wind_kt'] = pd.to_numeric(df['wind_kt'], errors='coerce')
        # Tạo cột cuong_do_bf cho logic icon
        df['cuong_do_bf'] = df['wind_kt'].apply(kt_to_bf)
    
    for col in ['lat', 'lon']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df.dropna(subset=['lat', 'lon', 'dt'])

def get_icon_name(row):
    """Logic xác định tên icon theo yêu cầu"""
    wind_speed = row.get('cuong_do_bf', 0)
    status = row.get('color_key', 'daqua')
    
    if pd.isna(wind_speed): return f"vungthap_{status}"
    if wind_speed < 6:      return f"vungthap_{status}"
    if wind_speed < 8:      return f"atnd_{status}"
    if wind_speed <= 11:    return f"bnd_{status}"
    return f"sieubao_{status}"

# --- 4. MAIN ---

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ CẤU HÌNH")
        default_file = "besttrack_capgio.xlsx"
        uploaded_file = st.file_uploader("File dữ liệu (.xlsx)", type=["xlsx"])
        data_source = uploaded_file if uploaded_file else (default_file if os.path.exists(default_file) else None)
        
        if not data_source: st.stop()
        df = load_data(data_source)
        if df is None or df.empty: st.stop()
            
        st.subheader("Lọc hiển thị")
        sel_storms = st.multiselect("Chọn Bão:", df['name'].unique(), default=df['name'].unique())
        
        if not df.empty and 'wind_kt' in df.columns:
            min_w, max_w = int(df['wind_kt'].min()), int(df['wind_kt'].max())
            w_range = st.slider("Cấp gió (kt):", min_w, max_w, (min_w, max_w))
            final_df = df[(df['name'].isin(sel_storms)) & (df['wind_kt'] >= w_range[0]) & (df['wind_kt'] <= w_range[1])]
        else: final_df = df

    # --- BẢN ĐỒ ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None)
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết').add_to(m)
    
    fg_past = folium.FeatureGroup(name="📉 Đường thực tế")
    fg_forecast = folium.FeatureGroup(name="🔮 Đường dự báo")
    fg_icons = folium.FeatureGroup(name="🌀 Biểu tượng Bão") # Layer riêng cho icon
    
    if not final_df.empty:
        for storm_name in sel_storms:
            storm_data = final_df[final_df['name'] == storm_name].sort_values('dt')
            if storm_data.empty: continue
            
            past_data = storm_data[storm_data['status'] != 'forecast']
            forecast_data = storm_data[storm_data['status'] == 'forecast']
            
            # Vẽ đường thực tế
            if not past_data.empty:
                folium.PolyLine(
                    past_data[['lat', 'lon']].values.tolist(), 
                    color='black', weight=2, opacity=0.8, tooltip=f"{storm_name}"
                ).add_to(fg_past)
                
            # Vẽ đường dự báo
            if not forecast_data.empty:
                if not past_data.empty:
                    conn = [[past_data.iloc[-1]['lat'], past_data.iloc[-1]['lon']], 
                            [forecast_data.iloc[0]['lat'], forecast_data.iloc[0]['lon']]]
                    folium.PolyLine(conn, color='red', weight=2, dash_array='5, 5').add_to(fg_forecast)
                folium.PolyLine(
                    forecast_data[['lat', 'lon']].values.tolist(), 
                    color='red', weight=2, dash_array='5, 5'
                ).add_to(fg_forecast)

            # VẼ ICON THAY CHO CHẤM MÀU
            for _, row in storm_data.iterrows():
                # 1. Lấy tên icon cơ sở (ví dụ: sieubao_daqua)
                icon_base_name = get_icon_name(row)
                
                # 2. Tạo đường dẫn file (ưu tiên .png, check thêm .PNG nếu cần)
                icon_path = os.path.join(ICON_DIR, f"{icon_base_name}.png")
                
                # Thông tin popup
                popup_content = f"""
                <div style='font-family:Arial; width:150px'>
                    <b>{row['name']}</b><br>
                    Time: {row['dt'].strftime('%d/%m %Hh')}<br>
                    Gió: {int(row.get('wind_kt',0))} kt (Cấp {int(row.get('cuong_do_bf',0))})
                </div>
                """

                # 3. Vẽ Marker
                if os.path.exists(icon_path):
                    # Kích thước icon: Bão lớn vẽ to hơn chút, hoặc để cố định (30,30)
                    icon_size = (35, 35) if 'sieubao' in icon_base_name else (25, 25)
                    
                    custom_icon = folium.CustomIcon(icon_path, icon_size=icon_size)
                    
                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        icon=custom_icon,
                        popup=folium.Popup(popup_content, max_width=200)
                    ).add_to(fg_icons)
                else:
                    # Fallback: Nếu không tìm thấy ảnh thì vẽ chấm tròn mặc định
                    color = '#808080' # Màu xám nếu thiếu icon
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']], radius=4,
                        color=color, fill=True, fill_opacity=1, popup=popup_content
                    ).add_to(fg_icons)

    fg_past.add_to(m)
    fg_forecast.add_to(m)
    fg_icons.add_to(m)
    
    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
