# -*- coding: utf-8 -*-
import os
import re  # <--- Thêm thư viện này để xử lý tìm số trong chuỗi
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io import shapereader as shpreader
from cartopy.feature import ShapelyFeature

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# Cấu hình
# ==============================================================================
FILE_PATH        = "besttrack_capgio.xlsx"
OUTPUT_FILE      = "mua_bao_2025_legend_sorted.png"
SHP_PATH         = "/home/lephong/ve_excel/vn/vn34tinh.shp"

YEAR_FILTER      = 2025

REQUIRED_COLS = {
    "tên bão": "name",
    "biển đông": "storm_no",
    "năm": "year",
    "tháng": "mon",
    "ngày": "day",
    "giờ": "hour",
    "vĩ độ": "lat",
    "kinh độ": "lon",
    "gió (kt)": "wind_kt",
    "Ngày - giờ": "ngay_gio",
}

# ==============================================================================
# Xử lý dữ liệu
# ==============================================================================
def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    rename_map = {k: v for k, v in REQUIRED_COLS.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    if all(c in df.columns for c in ['year', 'mon', 'day', 'hour']):
        df['dt'] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day, hour=df.hour), errors='coerce')
    elif 'ngay_gio' in df.columns:
        df['dt'] = pd.to_datetime(df['ngay_gio'], errors='coerce')
    
    df = df.dropna(subset=['lat', 'lon', 'dt'])
    
    if 'year' in df.columns:
        df = df[df['year'] == YEAR_FILTER]
        
    return df.sort_values(['name', 'dt'])

# Hàm hỗ trợ lấy số từ chuỗi "Bão số 10" để sắp xếp
def extract_storm_number(storm_no_str):
    if pd.isna(storm_no_str): return 999
    storm_no_str = str(storm_no_str)
    # Tìm các con số trong chuỗi
    match = re.search(r'(\d+)', storm_no_str)
    if match:
        return int(match.group(1)) # Trả về số nguyên (1, 2, 10...)
    return 999 # Nếu không có số (VD: ATNĐ) thì đẩy xuống cuối

# ==============================================================================
# Vẽ bản đồ
# ==============================================================================
def setup_map(extent):
    fig = plt.figure(figsize=(14, 10), dpi=300) # dpi 300 là đủ nét cho in ấn
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, linestyle="--", edgecolor='gray')
    
    xticks = np.arange(100, 148, 5)
    yticks = np.arange(5, 40, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.gridlines(xlocs=xticks, ylocs=yticks, linewidth=0.5, color="gray", alpha=0.3, linestyle="--")
    
    return fig, ax

def add_vietnam_shapefile(ax, shp_path):
    if os.path.exists(shp_path):
        reader = shpreader.Reader(shp_path)
        
        # --- CẤU HÌNH HỆ TỌA ĐỘ CHO SHAPEFILE ---
        # Nếu shapefile của bạn ĐÃ là WGS84 (lat/lon), dùng ccrs.PlateCarree()
        # Nếu shapefile là VN2000, Cartopy có thể không tự nhận diện được nếu thiếu file .prj
        # Tuy nhiên, thông thường ccrs.PlateCarree() hoạt động tốt với hầu hết shapefile địa lý.
        
        feat = ShapelyFeature(reader.geometries(), ccrs.PlateCarree()) # <--- CRS gốc của dữ liệu
        
        ax.add_feature(feat, facecolor='none', edgecolor='#333', linewidth=0.8, zorder=2)
    else:
        print(f"Cảnh báo: Không tìm thấy file shape tại {shp_path}")

def main():
    try:
        print("Đang đọc dữ liệu...")
        df = load_data(FILE_PATH)
        
        extent = [98, 125, 5, 25] # Đã điều chỉnh lại khung nhìn cho hợp lý với Biển Đông
        fig, ax = setup_map(extent)
        add_vietnam_shapefile(ax, SHP_PATH)
        
        places = {"Hà Nội": (105.8, 21.0), "TP.HCM": (106.7, 10.8), "Hoàng Sa": (111.5, 16.5), "Trường Sa": (114.0, 9.0)}
        for name, (lon, lat) in places.items():
            ax.text(lon, lat, name, transform=ccrs.PlateCarree(), fontsize=9, color='darkblue', weight='bold', ha='center',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

        # --- PHẦN SỬA ĐỔI QUAN TRỌNG: SẮP XẾP DANH SÁCH BÃO ---
        raw_names = df['name'].unique()
        
        # Tạo danh sách tạm chứa thông tin để sort
        storm_list = []
        for name in raw_names:
            # Lấy dòng đầu tiên của bão để tìm "Bão số X"
            first_row = df[df['name'] == name].iloc[0]
            s_no_txt = str(first_row.get('storm_no', ''))
            
            # Lấy số thứ tự (1, 2, 3...)
            sort_key = extract_storm_number(s_no_txt)
            
            storm_list.append({
                'name': name,
                'storm_no_text': s_no_txt,
                'sort_key': sort_key
            })
            
        # Sắp xếp danh sách dựa trên sort_key (số hiệu bão)
        storm_list.sort(key=lambda x: x['sort_key'])
        
        # Lấy lại danh sách tên đã sắp xếp
        sorted_unique_storms = [item['name'] for item in storm_list]
        
        # Tạo màu
        colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_unique_storms)))
        
        legend_handles = []
        print(f"Bắt đầu vẽ {len(sorted_unique_storms)} cơn bão (đã sắp xếp)...")
        
        for i, storm_name in enumerate(sorted_unique_storms):
            sub = df[df['name'] == storm_name].sort_values('dt')
            if sub.empty: continue
            
            # Lấy text hiển thị từ danh sách đã chuẩn bị
            storm_info = next(item for item in storm_list if item['name'] == storm_name)
            storm_no_text = storm_info['storm_no_text']
            if storm_no_text == 'nan': storm_no_text = ''
            
            if storm_no_text:
                legend_label = f"{storm_no_text}: {storm_name}"
            else:
                legend_label = storm_name
                
            color = colors[i]
            
            # 1. Vẽ đường đi
            line, = ax.plot(sub['lon'], sub['lat'], color=color, linewidth=2.5, 
                            transform=ccrs.PlateCarree(), label=legend_label, zorder=5)
            legend_handles.append(line)
            
            # 2. Đánh dấu tên tại điểm khởi đầu
            start_pt = sub.iloc[0]
            ax.text(start_pt['lon'], start_pt['lat'], storm_no_text, 
                    transform=ccrs.PlateCarree(), fontsize=9, color=color, weight='bold',
                    ha='right', va='bottom', zorder=10,
                    path_effects=[path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
            
            ax.scatter(start_pt['lon'], start_pt['lat'], color=color, s=30, zorder=6, transform=ccrs.PlateCarree())

        # Tạo bảng chú thích (Legend)
        leg = ax.legend(handles=legend_handles, loc='upper right', 
                        bbox_to_anchor=(0.99, 0.99),
                        title="CHÚ GIẢI", fontsize=10, title_fontsize=12,
                        facecolor='white', framealpha=0.9, edgecolor='gray')
        
        ax.set_title(f"MÙA BÃO TRÊN BIỂN ĐÔNG NĂM {YEAR_FILTER}", fontsize=15, weight='bold', color='#003366', pad=15)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight') # dpi 300 là chuẩn in ấn
        print(f"XONG! Đã lưu bản đồ tại: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()