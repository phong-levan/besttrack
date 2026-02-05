import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# Các thư viện sau được giữ lại nhưng phần code liên quan đến Convex Hull đã bị comment
# from cartopy import geodesic
# from scipy.spatial import ConvexHull
# from shapely.geometry import Polygon
# from matplotlib.patches import Rectangle 

# Bỏ qua cảnh báo của NumPy nếu nó không ảnh hưởng đến chương trình
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# Cấu hình
# ==============================================================================
FILE_PATH = "/home/lephong/besttrack/besttrack.xlsx"
ICON_FOLDER_PATH = '/home/lephong/besttrack/icon'
OUTPUT_FILE = "/home/lephong/besttrack/bao_so_10_2025/2025092506Z.png"

# Định nghĩa các cột cần thiết (tiếng Việt không dấu và tên biến chuẩn Python)
REQUIRED_COLS = {
    'Thời điểm': 'thoi_diem',
    'lat': 'lat',
    'lon': 'lon',
    'trạng thái': 'trang_thai',
    'cường độ (cấp BF)': 'cuong_do_bf',
    'bán kính tâm (km)': 'ban_kinh_tam',
    'bán kính gió mạnh cấp 10 (km)': 'ban_kinh_gio_10',
    'bán kính gió mạnh cấp 6 (km)': 'ban_kinh_gio_6',
    'Vmax (km/h)': 'vmax',
    'Pmin (mb)': 'pmin',
    'Ngày - giờ': 'ngay_gio'
}

ICON_PATHS = {
    "vungthap_daqua": os.path.join(ICON_FOLDER_PATH, 'vungthapdaqua.png'),
    "atnd_daqua": os.path.join(ICON_FOLDER_PATH, 'atnddaqua.PNG'),
    "bnd_daqua": os.path.join(ICON_FOLDER_PATH, 'bnddaqua.PNG'),
    "sieubao_daqua": os.path.join(ICON_FOLDER_PATH, 'sieubaodaqua.PNG'),
    "vungthap_dubao": os.path.join(ICON_FOLDER_PATH, 'vungthapdubao.png'),
    "atnd_dubao": os.path.join(ICON_FOLDER_PATH, 'atnd.PNG'),
    "bnd_dubao": os.path.join(ICON_FOLDER_PATH, 'bnd.PNG'),
    "sieubao_dubao": os.path.join(ICON_FOLDER_PATH, 'sieubao.PNG')
}

# ==============================================================================
# Đọc và Tiền xử lý Dữ liệu Bão
# ==============================================================================
def load_and_preprocess_data(file_path, required_cols):
    """Đọc tệp Excel và tiền xử lý dữ liệu."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp dữ liệu tại {file_path}")

    df_excel = pd.read_excel(file_path)
    
    # Đổi tên cột để dễ sử dụng (từ tiếng Việt có dấu sang tiếng Anh/viết tắt chuẩn)
    df = df_excel.rename(columns={k: v for k, v in required_cols.items() if k in df_excel.columns})

    # Kiểm tra và thêm các cột bị thiếu với giá trị NaN
    for original_col, new_col in required_cols.items():
        if original_col not in df_excel.columns:
            print(f"CẢNH BÁO: Cột '{original_col}' không có trong file Excel. Sử dụng giá trị NaN cho '{new_col}'.")
            df[new_col] = np.nan
            
    # Xóa các hàng không có lat/lon
    df = df.dropna(subset=["lat", "lon"]).copy()

    # Thêm cột phân loại bão và màu sắc/trạng thái
    if 'trang_thai' in df.columns:
        df["cat"] = df["trang_thai"].apply(lambda x: 'STY' if "SIEU BAO" in str(x).upper() else 'BND')
    else:
        df["cat"] = 'BND'

    if 'thoi_diem' in df.columns:
        df["color_key"] = df["thoi_diem"].apply(lambda x: 'daqua' if "quá khứ" in str(x).lower() else 'dubao')
    else:
        df["color_key"] = 'dubao'
        
    # Tạo tên icon dựa trên cường độ gió (cấp BF) và trạng thái (quá khứ/dự báo)
    def get_icon_name(row):
        wind_speed = row['cuong_do_bf']
        status = row['color_key']
        if pd.isna(wind_speed):
            return f"atnd_{status}" # Mặc định là áp thấp nhiệt đới
        elif wind_speed < 8:
            return f"atnd_{status}"
        elif 8 <= wind_speed <= 11:
            return f"bnd_{status}"
        else: # wind_speed >= 12
            return f"sieubao_{status}"

    df['icon_name'] = df.apply(get_icon_name, axis=1)

    return df

# ==============================================================================
# Cấu hình Bản đồ (Figure/Axes)
# ==============================================================================
def setup_map(extent):
    """Cài đặt đối tượng figure và axes cho bản đồ Cartopy."""
    fig = plt.figure(figsize=(12, 9), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Thêm các đối tượng địa lý
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#d6efff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3, linestyle="--", edgecolor='gray', zorder=1.5)

    # Thiết lập lưới tọa độ
    xticks = np.arange(100, 140, 5)
    yticks = np.arange(5, 35, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.gridlines(xlocs=xticks, ylocs=yticks, 
                 linewidth=0.5, color="k", alpha=0.35, linestyle="--", draw_labels=False, zorder=1)
    
    return fig, ax

# ==============================================================================
# Vẽ đường đi và Icon bão
# ==============================================================================
def plot_storm_track_and_icons(ax, df, icon_paths):
    """Vẽ đường đi bão và các icon trạng thái."""
    # Vẽ đường đi bão
    storm_path = df[["lon", "lat"]].values
    if len(storm_path) > 1:
        storm_path_lines = mlines.Line2D(storm_path[:, 0], storm_path[:, 1], 
                                         color='black', linestyle='-', linewidth=0.5, 
                                         transform=ccrs.PlateCarree(), zorder=5)
        ax.add_artist(storm_path_lines)
    
    # Tải trước các icon để tái sử dụng
    image_cache = {}
    for name, path in icon_paths.items():
        if os.path.exists(path):
            image_cache[name] = plt.imread(path)
        else:
            print(f"CẢNH BÁO: Không tìm thấy tệp icon tại '{path}'.")

    # Vẽ icon (tối ưu: tránh lặp qua df bằng iterrows nếu có thể, nhưng OffsetImage/AnnotationBbox
    # hiện tại cần lặp qua từng điểm)
    for i, row in df.iterrows():
        icon_name = row['icon_name']
        
        if icon_name in image_cache:
            img = image_cache[icon_name]
            oi = OffsetImage(img, zoom=0.25)
            ab = AnnotationBbox(oi, (row["lon"], row["lat"]), 
                                frameon=False, xycoords='data', zorder=12)
            ax.add_artist(ab)
        else:
            # Vẽ điểm thay thế nếu icon không tồn tại
            status = row['color_key']
            color = 'black' if status == 'daqua' else 'red'
            ax.plot(row["lon"], row["lat"], 'o', color=color, markersize=5, 
                    transform=ccrs.PlateCarree(), zorder=12)

# ==============================================================================
# Thêm các Nhãn Địa lý
# ==============================================================================
def add_place_labels(ax):
    """Thêm nhãn các địa điểm quan trọng."""
    place_labels = {
        "Hà Nội": (105.8, 21.0), "Đà Nẵng": (108.2, 16.0), "TP.HCM": (106.7, 10.8),
        "DK.Hoàng Sa": (111.7, 16.0), "DK.Trường Sa": (113.8, 8.5)
    }
    for name, (lon, lat) in place_labels.items():
        ax.text(lon, lat, name, fontsize=3, color='darkblue', weight='bold', 
                ha='center', va='center', transform=ccrs.PlateCarree(), zorder=10)

# ==============================================================================
# Thêm Bảng Thông tin Dự báo
# ==============================================================================
def add_info_table(ax, df_forecast, icon_folder_path):
    """Vẽ bảng thông tin dự báo (hiện tại và tương lai) lên bản đồ."""
    box_lon_min, box_lon_max = 122, 135
    box_lat_min, box_lat_max = 21.5, 29.0
    
    # Lọc dữ liệu: chỉ giữ lại các dòng không phải là "quá khứ"
    df_hientai_dubao = df_forecast[df_forecast['color_key'] != 'daqua'].copy()

    # Vẽ hình chữ nhật nền
    ax.add_patch(plt.Rectangle((box_lon_min, box_lat_min), box_lon_max - box_lon_min, box_lat_max - box_lat_min, 
                                facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.7, 
                                transform=ccrs.PlateCarree(), zorder=20))

    # Tiêu đề bảng
    ax.text(box_lon_min + 6.5, box_lat_max - 0.8, "TIN BÃO GẦN BIỂN ĐÔNG", 
            fontsize=12, weight='bold', ha='center', va='top', transform=ccrs.PlateCarree(), zorder=21)
    ax.text(box_lon_min + 6.5, box_lat_max - 1.8, "Tin phát lúc: 14 giờ 00 phút, 25/09/2025", 
            fontsize=10, ha='center', va='top', transform=ccrs.PlateCarree(), zorder=21)

    # Định vị các cột
    col_pos = { 
        "Ngay-gio": box_lon_min + 1.8, 
        "Kinh do": box_lon_min + 4.5, 
        "Vi do": box_lon_min + 6.5, 
        "Cap gio": box_lon_min + 8.5, 
        "Pmin": box_lon_min + 11.2 
    } 

    # Tiêu đề cột
    header_y = box_lat_max - 2.6
    headers = [("Ngay-gio", "Ngày - giờ"), ("Kinh do", "Kinh độ"), ("Vi do", "Vĩ độ"), 
               ("Cap gio", "Cấp gió"), ("Pmin", "Pmin(mb)")]
    for key, text in headers:
        ax.text(col_pos[key], header_y, text, fontsize=10, weight='bold', ha='center', va='top', 
                transform=ccrs.PlateCarree(), zorder=21)
    
    # Dữ liệu
    y_offset = box_lat_max - 3.4
    row_height = 0.6
    
    for _, row in df_hientai_dubao.iterrows():
        pmin = row.get('pmin')
        pmin_str = f"{int(pmin)}" if pd.notna(pmin) and pd.api.types.is_number(pmin) else "N/A"
        
        data_points = [
            (col_pos["Ngay-gio"], str(row['ngay_gio'])),
            (col_pos["Kinh do"], f"{row['lon']:.1f}E"),
            (col_pos["Vi do"], f"{row['lat']:.1f}N"),
            (col_pos["Cap gio"], f"Cấp {int(row['cuong_do_bf'])}"),
            (col_pos["Pmin"], pmin_str)
        ]

        for x_pos, text in data_points:
            ax.text(x_pos, y_offset, text, fontsize=10, ha='center', va='top', 
                    transform=ccrs.PlateCarree(), zorder=21)
        
        y_offset -= row_height

    # Thêm chú thích bằng hình ảnh
    chuthich_path = os.path.join(icon_folder_path, 'chugiai.png')
    if os.path.exists(chuthich_path):
        legend_img = plt.imread(chuthich_path)
        oi_legend = OffsetImage(legend_img, zoom=0.25)
        # Vị trí chú thích được căn chỉnh tương đối với vị trí cuối cùng của bảng
        ab_legend = AnnotationBbox(oi_legend, (box_lon_min + 6.4, y_offset + 0.05), 
                                   xycoords='data', box_alignment=(0.5, 1.2), frameon=False, zorder=22)
        ax.add_artist(ab_legend)
    else:
        print(f"CẢNH BÁO: Không tìm thấy tệp chú thích tại '{chuthich_path}'.")

# ==============================================================================
# Phần Chính: Thực thi Chương trình
# ==============================================================================
def main():
    try:
        # 1. Đọc và tiền xử lý dữ liệu
        df = load_and_preprocess_data(FILE_PATH, REQUIRED_COLS)
        
        # 2. Cài đặt bản đồ
        map_extent = [97, 137, 3, 30] # [lon_min, lon_max, lat_min, lat_max]
        fig, ax = setup_map(map_extent)

        # 3. Vẽ vùng gió mạnh (Phần này đã bị comment trong code gốc nên giữ nguyên trạng thái)
        # plot_wind_radii(ax, df) # Hàm này cần được định nghĩa lại nếu muốn dùng

        # 4. Vẽ đường đi và Icon bão
        plot_storm_track_and_icons(ax, df, ICON_PATHS)

        # 5. Thêm các nhãn địa lý
        add_place_labels(ax)

        # 6. Thêm Bảng Thông tin Dự báo
        add_info_table(ax, df, ICON_FOLDER_PATH)

        # 7. Lưu và hiển thị
        ax.set_title("Dự báo quỹ đạo và vùng ảnh hưởng của bão", fontsize=11, weight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=1000)
        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()