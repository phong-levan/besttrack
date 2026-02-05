# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Hình học để hợp nhất vùng gió
from cartopy import geodesic
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# Cấu hình
# ==============================================================================
FILE_PATH = "besttrack.xlsx"
ICON_FOLDER_PATH = 'icon'
OUTPUT_FILE = "bao_so_14_2025/2025111003Z.png"

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

# Màu vùng gió + đường đi
COL_R6   = "#FFC0CB"
COL_R10  = "#FF6347"
COL_RC   = "#90EE90"
COL_TRK_PAST = "black"
COL_TRK_CURF = "black"

# ==============================================================================
# Tiện ích thời gian / lọc
# ==============================================================================
def to_datetime_series(df):
    cols = {c.lower().strip(): c for c in df.columns}
    if 'ngay_gio' in df.columns:
        dt = pd.to_datetime(df['ngay_gio'], dayfirst=True, errors="coerce")
        if dt.notna().any():
            return dt
    for name in ["ngày-giờ","ngày giờ","datetime","date_time","time","date","ngay - gio"]:
        if name in cols:
            return pd.to_datetime(df[cols[name]], dayfirst=True, errors="coerce")
    return pd.to_datetime(pd.Series(range(len(df))), unit="h")

def get_time_window(sheets, main_df):
    for key, fdf in sheets.items():
        if str(key).strip().lower() == "filter":
            cols = {c.lower().strip(): c for c in fdf.columns}
            s = next((cols.get(k) for k in ["start","bắt đầu","from","từ"] if k in cols), None)
            e = next((cols.get(k) for k in ["end","kết thúc","to","đến"] if k in cols), None)
            if s and e and fdf[s].notna().any() and fdf[e].notna().any():
                try:
                    return (
                        pd.to_datetime(fdf[s].dropna().iloc[0], dayfirst=True, errors="raise"),
                        pd.to_datetime(fdf[e].dropna().iloc[0], dayfirst=True, errors="raise"),
                    )
                except Exception:
                    pass
    cols = {c.lower().strip(): c for c in main_df.columns}
    s = next((cols.get(k) for k in ["start","bắt đầu","from","từ"] if k in cols), None)
    e = next((cols.get(k) for k in ["end","kết thúc","to","đến"] if k in cols), None)
    if s and e and main_df[s].notna().any() and main_df[e].notna().any():
        try:
            return (
                pd.to_datetime(main_df[s].dropna().iloc[0], dayfirst=True, errors="raise"),
                pd.to_datetime(main_df[e].dropna().iloc[0], dayfirst=True, errors="raise"),
            )
        except Exception:
            pass
    for colkey in ["keep","giữ","giu","use","selected"]:
        if colkey in cols:
            kept = main_df[main_df[cols[colkey]].astype(str).str.strip().isin(["1","True","true","TRUE"])]
            if not kept.empty:
                dt2 = to_datetime_series(kept)
                return dt2.min(), dt2.max()
    return None, None

# ==============================================================================
# Đọc & tiền xử lý
# ==============================================================================
def load_and_preprocess_data(file_path, required_cols):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp dữ liệu tại {file_path}")

    sheets = pd.read_excel(file_path, sheet_name=None)
    
    all_dfs = []
    for name, sdf in sheets.items():
        cols = [c.lower().strip() for c in sdf.columns]
        if "lat" in cols and "lon" in cols:
            all_dfs.append(sdf)
    
    if not all_dfs:
         raise ValueError("Không tìm thấy sheet nào chứa cột 'lat' và 'lon' trong file Excel.")
    
    df_excel = pd.concat(all_dfs, ignore_index=True)

    df = df_excel.rename(columns={k: v for k, v in required_cols.items() if k in df_excel.columns})

    for original_col, new_col in required_cols.items():
        if original_col not in df_excel.columns and new_col not in df.columns:
            print(f"CẢNH BÁO: Thiếu cột '{original_col}' → tạo '{new_col}'=NaN.")
            df[new_col] = np.nan

    df = df.dropna(subset=["lat", "lon"]).copy()
    if 'Ngày - giờ' in df_excel.columns:
        df['ngay_gio'] = df_excel['Ngày - giờ'].astype(str)
    
    df["dt"] = to_datetime_series(df)
    df = df.dropna(subset=['dt'])

    if 'trang_thai' in df.columns:
        df["cat"] = df["trang_thai"].apply(lambda x: 'STY' if "SIEU BAO" in str(x).upper() else 'BND')
    else:
        df["cat"] = 'BND'

    if 'thoi_diem' in df.columns:
        df["color_key"] = df["thoi_diem"].apply(lambda x: 'daqua' if "quá khứ" in str(x).lower() else 'dubao')
    else:
        df["color_key"] = 'dubao'

    def get_icon_name(row):
        wind_speed = row['cuong_do_bf']
        status = row['color_key']
        if pd.isna(wind_speed): return f"vungthap_{status}"
        if wind_speed < 6:      return f"vungthap_{status}"
        if wind_speed < 8:      return f"atnd_{status}"
        if wind_speed <= 11:    return f"bnd_{status}"
        return f"sieubao_{status}"
        
    df['icon_name'] = df.apply(get_icon_name, axis=1)

    start, end = get_time_window(sheets, df)
    return df.sort_values("dt").reset_index(drop=True), (start, end)

# ==============================================================================
# Map
# ==============================================================================
def setup_map(extent):
    fig = plt.figure(figsize=(12, 9), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#d6efff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3, linestyle="--", edgecolor='gray', zorder=1.5)
    xticks = np.arange(95, 155, 5)
    yticks = np.arange(5, 35, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.gridlines(xlocs=xticks, ylocs=yticks, linewidth=0.5, color="k", alpha=0.35, linestyle="--", draw_labels=False)
    return fig, ax

# ==============================================================================
# Hình học
# ==============================================================================
def geodesic_circle_polygon(lon, lat, radius_km, n_samples=360):
    if pd.isna(radius_km) or radius_km is None or radius_km <= 0:
        return None
    arr = geodesic.Geodesic().circle(lon=lon, lat=lat, radius=radius_km*1000.0, n_samples=n_samples)
    return Polygon(arr)

def union_of_circles(lons, lats, radii_km, n_samples=360):
    polys = []
    for lo, la, r in zip(lons, lats, radii_km):
        p = geodesic_circle_polygon(lo, la, r, n_samples=n_samples)
        if p is not None and p.is_valid:
            polys.append(p)
    if not polys:
        return None
    return unary_union(polys)

def iter_geoms(geom):
    if geom is None: return []
    if isinstance(geom, Polygon): return [geom]
    if isinstance(geom, MultiPolygon): return list(geom.geoms)
    try: return list(geom.geoms)
    except Exception: return [geom]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    from math import radians, sin, cos, asin, sqrt
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi  = radians(lat2 - lat1)
    dlamb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlamb/2)**2
    return 2*R*asin(sqrt(a))

def densify_track(lons, lats, *radii_arrays, step_km=25):
    nums = len(radii_arrays)
    outs = [[] for _ in range(nums)]
    new_lo, new_la = [], []
    for i in range(len(lons)-1):
        lo1, la1, lo2, la2 = lons[i], lats[i], lons[i+1], lats[i+1]
        d = haversine_km(la1, lo1, la2, lo2)
        n = max(1, int(np.ceil(d / step_km)))
        for j in range(n):
            t = j / n
            new_lo.append(lo1*(1-t) + lo2*t)
            new_la.append(la1*(1-t) + la2*t)
            for k in range(nums):
                outs[k].append(radii_arrays[k][i]*(1-t) + radii_arrays[k][i+1]*t)
    new_lo.append(lons[-1]); new_la.append(lats[-1])
    for k in range(nums): outs[k].append(radii_arrays[k][-1])
    return np.array(new_lo), np.array(new_la), [np.array(x) for x in outs]

# ==============================================================================
# Vùng gió (ĐÃ SỬA LỖI HOÀN CHỈNH)
# ==============================================================================
def plot_wind_corridors(ax, df, time_window=None):
    # Lọc các điểm dữ liệu dự báo và hiện tại
    sub = df[df['color_key'] != 'daqua'].copy()
    if time_window and 'dt' in sub.columns:
        start, end = time_window
        if start is not None and end is not None:
            sub = sub[(sub['dt'] >= start) & (sub['dt'] <= end)].copy()
    if sub.empty:
        print("[INFO] Không có điểm hiện tại/dự báo để vẽ vùng gió.")
        return

    # Lấy dữ liệu tọa độ và bán kính
    lons = sub['lon'].astype(float).to_numpy()
    lats = sub['lat'].astype(float).to_numpy()
    r6   = pd.to_numeric(sub.get('ban_kinh_gio_6', np.nan), errors="coerce").fillna(0).to_numpy(float)
    r10  = pd.to_numeric(sub.get('ban_kinh_gio_10', np.nan), errors="coerce").fillna(0).to_numpy(float)
    rc   = pd.to_numeric(sub.get('ban_kinh_tam',   np.nan), errors="coerce").fillna(0).to_numpy(float)

    # Nội suy để lấp các giá trị bán kính bị thiếu
    def _interp(a):
        return pd.Series(a).replace(0, np.nan).interpolate(limit_direction="both").fillna(0).to_numpy()
    r6, r10, rc = _interp(r6), _interp(r10), _interp(rc)

    # Làm mịn đường đi và bán kính để vùng gió mượt hơn
    if len(lons) >= 2:
        lons_d, lats_d, (r6_d, r10_d, rc_d) = densify_track(lons, lats, r6, r10, rc, step_km=25)
    else:
        lons_d, lats_d, r6_d, r10_d, rc_d = lons, lats, r6, r10, rc

    # >>> ĐÂY LÀ PHẦN QUAN TRỌNG BỊ THIẾU DẪN ĐẾN LỖI NAMEERROR <<<
    # *** BƯỚC 1: TẠO CÁC VÙNG HÌNH HỌC GỐC (CHƯA BỊ CẮT) ***
    swath_r6  = union_of_circles(lons_d, lats_d, r6_d,  n_samples=360)
    swath_r10 = union_of_circles(lons_d, lats_d, r10_d, n_samples=360)
    swath_rc  = union_of_circles(lons_d, lats_d, rc_d,  n_samples=360)

    # *** BƯỚC 2: TẠO CÁC VÙNG HÌNH HỌC KHÔNG CHỒNG CHÉO ***
    # Tạo vùng màu đỏ (vùng cấp 10 trừ đi vùng tâm bão)
    plot_swath_r10 = None
    if swath_r10:
        if swath_rc:
            plot_swath_r10 = swath_r10.difference(swath_rc)
        else:
            plot_swath_r10 = swath_r10

    # Tạo vùng màu hồng (vùng cấp 6 trừ đi tất cả các vùng bên trong)
    plot_swath_r6 = None
    if swath_r6:
        # Gộp vùng đỏ và vùng tâm bão lại làm một để trừ
        inner_swath_to_subtract = unary_union([s for s in [swath_r10, swath_rc] if s is not None and not s.is_empty])
        if inner_swath_to_subtract:
            plot_swath_r6 = swath_r6.difference(inner_swath_to_subtract)
        else:
            plot_swath_r6 = swath_r6

    # *** BƯỚC 3: VẼ CÁC VÙNG ĐÃ ĐƯỢC XỬ LÝ LÊN BẢN ĐỒ ***
    # Vẽ theo thứ tự từ ngoài vào trong: Hồng -> Đỏ -> Xanh
    if plot_swath_r6:
        for g in iter_geoms(plot_swath_r6):
            ax.add_geometries([g], crs=ccrs.PlateCarree(), facecolor=COL_R6,  edgecolor='none', alpha=0.50, zorder=2.5)
            
    if plot_swath_r10:
        for g in iter_geoms(plot_swath_r10):
            ax.add_geometries([g], crs=ccrs.PlateCarree(), facecolor=COL_R10, edgecolor='none', alpha=0.60, zorder=2.8)
            
    if swath_rc:
        for g in iter_geoms(swath_rc):
            ax.add_geometries([g], crs=ccrs.PlateCarree(), facecolor=COL_RC,  edgecolor='none', alpha=0.70, zorder=3.0)

# ==============================================================================
# Đường đi
# ==============================================================================
def plot_tracks(ax, df):
    df_past = df[df['color_key'] == 'daqua'].copy().sort_values('dt')
    df_curf = df[df['color_key'] != 'daqua'].copy().sort_values('dt')

    if not df_past.empty:
        lons = df_past['lon'].astype(float).to_numpy()
        lats = df_past['lat'].astype(float).to_numpy()
        ax.plot(lons, lats, color=COL_TRK_PAST, linewidth=1.4, zorder=5, transform=ccrs.PlateCarree())

    if not df_curf.empty:
        lons = df_curf['lon'].astype(float).to_numpy()
        lats = df_curf['lat'].astype(float).to_numpy()
        ax.plot(lons, lats, color=COL_TRK_CURF, linewidth=1.8, linestyle="-", zorder=6, transform=ccrs.PlateCarree())
    
    if not df_past.empty and not df_curf.empty:
        last_past_lon = df_past['lon'].iloc[-1]
        last_past_lat = df_past['lat'].iloc[-1]
        first_curf_lon = df_curf['lon'].iloc[0]
        first_curf_lat = df_curf['lat'].iloc[0]

        ax.plot([last_past_lon, first_curf_lon], [last_past_lat, first_curf_lat],
                color=COL_TRK_CURF, linewidth=1.8, linestyle="-", zorder=5.5, 
                transform=ccrs.PlateCarree())

# ==============================================================================
# Icon tâm bão
# ==============================================================================
def plot_storm_track_and_icons(ax, df, icon_paths):
    image_cache = {}
    for name, path in icon_paths.items():
        if os.path.exists(path):
            image_cache[name] = plt.imread(path)
        else:
            print(f"CẢNH BÁO: Không tìm thấy tệp icon tại '{path}'.")

    for _, row in df.iterrows():
        icon_name = row['icon_name']
        if icon_name in image_cache:
            img = image_cache[icon_name]
            
            if "vungthap" in icon_name:
                zoom_level = 0.08
            else:
                zoom_level = 0.25

            oi = OffsetImage(img, zoom=zoom_level)
            ab = AnnotationBbox(oi, (row["lon"], row["lat"]), frameon=False,
                                xycoords='data', zorder=12, box_alignment=(0.5,0.5))
            ax.add_artist(ab)

# ==============================================================================
# Nhãn địa lý
# ==============================================================================
def add_place_labels(ax):
    place_labels = {
        "Hà Nội": (105.8, 21.0), "Đà Nẵng": (108.2, 16.0), "TP.HCM": (106.7, 10.8),
        "DK.Hoàng Sa": (111.7, 16.0), "DK.Trường Sa": (113.8, 8.5)
    }
    for name, (lon, lat) in place_labels.items():
        ax.text(lon, lat, name, fontsize=6, color='darkblue', weight='bold',
                ha='center', va='center', transform=ccrs.PlateCarree(), zorder=10,
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                                path_effects.Normal()])

# ==============================================================================
# Bảng thông tin (GIÃN ĐỀU THEO CỘT VÀ HÀNG, CHÚ THÍCH CÁCH ĐÁY 1.0)
# ==============================================================================
def add_info_table(ax, df_forecast, icon_folder_path, time_window=None):
    # --- BƯỚC 1: GIỮ NGUYÊN KÍCH THƯỚC KHUNG CỐ ĐỊNH ---
    box_lon_min, box_lon_max = 126.5, 152.3
    # Đáy khung cũ là 21.0
    box_lat_min, box_lat_max = 21.0, 32.5 
    
    # [THAY ĐỔI ĐỂ CHỨA CHÚ THÍCH]: Nếu bạn muốn Chú thích nằm bên dưới, 
    # bạn cần đảm bảo ax (trục) có thể hiển thị nó.
    # Trong code hiện tại, chúng ta sẽ coi box_lat_min là vị trí bắt đầu
    # của khu vực Chú thích.

    # --- BƯỚC 2: LỌC DỮ LIỆU ---
    sub = df_forecast[df_forecast['color_key'] != 'daqua'].copy()
    if time_window and 'dt' in sub.columns:
        start, end = time_window
        if start is not None and end is not None:
            sub = sub[(sub['dt'] >= start) & (sub['dt'] <= end)].copy()
            
    num_rows = len(sub)
    if num_rows == 0:
        print("[INFO] Không có dữ liệu dự báo để hiển thị trong bảng.")
        return

    # --- BƯỚC 3: VẼ KHUNG VÀ TIÊU ĐỀ ---
    # KHUNG BẢNG CHÍNH VẪN DÙNG box_lat_min = 21.0
    ax.add_patch(plt.Rectangle((box_lon_min, box_lat_min),
                               box_lon_max - box_lon_min, box_lat_max - box_lat_min,
                               facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.7,
                               transform=ccrs.PlateCarree(), zorder=20))
    
    box_center_x = box_lon_min + (box_lon_max - box_lon_min) / 2
    
    # Tiêu đề
    ax.text(box_center_x, box_lat_max - 0.5, "TIN BÃO TRÊN BIỂN ÐÔNG (Con bão số 14)",
            fontsize=14, weight='bold', ha='center', va='top', transform=ccrs.PlateCarree(), zorder=21)
    # Tiêu đề phụ
    ax.text(box_center_x, box_lat_max - 2.0, "Tin phát lúc: 10 giờ 30 phút, 10/11/2025",
            fontsize=9, ha='center', va='top', transform=ccrs.PlateCarree(), zorder=21)

    # --- BƯỚC 4: CĂN CHỈNH CỘT (GIỮ NGUYÊN) ---
    total_width = box_lon_max - box_lon_min
    margin_left = 0.5
    col_width_ngay = total_width * 0.2 
    remaining_width = total_width - col_width_ngay - margin_left
    col_width_other = remaining_width / 4
    pos_kinhdo = box_lon_min + margin_left + col_width_ngay + (col_width_other / 2)
    pos_vido   = pos_kinhdo + col_width_other
    pos_capgio = pos_vido + col_width_other
    pos_pmin   = pos_capgio + col_width_other

    col_pos = {
        "Ngay-gio": box_lon_min + margin_left, "Kinh do":  pos_kinhdo,
        "Vi do":    pos_vido, "Cap gio":  pos_capgio, "Pmin": pos_pmin,
    }
    col_align = {
        "Ngay-gio": 'left', "Kinh do": 'center', "Vi do": 'center',
        "Cap gio": 'center', "Pmin": 'center'
    }

    # --- BƯỚC 5: VẼ HEADER BẢNG (CỐ ĐỊNH) ---
    header_y = box_lat_max - 2.8 # Vị trí Y cố định cho header
    headers = [("Ngay-gio", "Ngày - giờ"), ("Kinh do", "Kinh độ"),
               ("Vi do", "Vĩ độ"), ("Cap gio", "Cấp gió"), ("Pmin", "Pmin(mb)")]
    for key, text in headers:
        ax.text(col_pos[key], header_y, text, fontsize=10, weight='bold', ha=col_align[key], va='top',
                transform=ccrs.PlateCarree(), zorder=21)

    # --- BƯỚC 6: TÍNH TOÁN VỊ TRÍ VÀ CHIỀU CAO HÀNG (GIÃN ĐỀU THEO HÀNG) ---
    
    # 1. Vị trí Y bắt đầu cho hàng dữ liệu đầu tiên (Đỉnh của hàng)
    y_data_start_top = header_y - 0.8

    # 2. Chiều cao ƯỚC TÍNH của Chú Thích (ảnh chuthich.PNG, zoom 0.2)
    legend_height_approx = 2.8 
    
    # 3. [SỬA] Khoảng cách từ ĐÁY ẢNH Chú thích đến ĐÁY KHUNG BẢNG (21.0)
    # Nếu muốn đẩy xuống dưới chân khung, ta sẽ đặt ảnh cách đáy khung 0.5 (khoảng cách an toàn)
    # Và đặt Chú thích nằm NGOÀI KHUNG.
    
    # Khoảng cách 1.0 là khoảng cách từ chân ảnh Chú thích đến tọa độ Y nào đó.
    # Đáy khung bảng là box_lat_min = 21.0.
    # Giả sử bạn muốn đáy ảnh Chú thích cách 21.0 là 1.0 (tức là tại Y=20.0).
    padding_bottom_from_box_lat_min = 5.0 
    y_legend_bottom = box_lat_min - padding_bottom_from_box_lat_min # = 21.0 - 6.0 = 15.0

    # 4. Vị trí Y cho TÂM của Chú Thích
    y_legend_center = y_legend_bottom + (legend_height_approx / 2)

    # 5. Vị trí Y cho ĐỈNH của khu vực Chú Thích
    # Vị trí này chính là y_legend_top_area = y_legend_bottom + legend_height_approx
    # Nhưng vì Chú thích nằm bên ngoài khung bảng (dưới 21.0), nên ta không cần dùng nó để tính không gian
    # cho bảng dữ liệu. Ta dùng đáy khung bảng (21.0) làm ranh giới.
    y_table_bottom_limit = box_lat_min 

    # 6. Tính toán không gian còn lại cho TẤT CẢ các hàng dữ liệu (ranh giới là đáy khung bảng)
    available_data_row_space = y_data_start_top - y_table_bottom_limit 
    
    # 7. TÍNH CHIỀU CAO ĐỘNG CỦA MỖI HÀNG (GIÃN ĐỀU THEO HÀNG)
    if num_rows > 0:
        # Chia đều không gian và đảm bảo chiều cao tối thiểu là 0.1
        row_height = max(0.1, available_data_row_space / num_rows) 
    else:
        row_height = 0 

    # --- BƯỚC 7: VẼ CÁC DÒNG DỮ LIỆU (CĂN GIỮA DÒNG ĐÃ GIÃN ĐỀU) ---
    # Khởi tạo y_offset tại TÂM của hàng đầu tiên
    y_offset = y_data_start_top - (row_height / 2) 
    
    for _, row in sub.iterrows():
        pmin = row.get('pmin')
        pmin_str = f"{int(pmin)}" if pd.notna(pmin) and pd.api.types.is_number(pmin) else "N/A"
        lvl = row.get('cuong_do_bf')
        lvl_str = "N/A" if pd.isna(lvl) else f"Cấp {int(lvl)}"
        ngay_gio_str = str(row.get('ngay_gio', ''))

        data_points = [
            (col_pos["Ngay-gio"], ngay_gio_str),
            (col_pos["Kinh do"], f"{float(row['lon']):.1f}E"),
            (col_pos["Vi do"],   f"{float(row['lat']):.1f}N"),
            (col_pos["Cap gio"], lvl_str),
            (col_pos["Pmin"],    pmin_str)
        ]
        
        # Vẽ từng cột của dòng, căn vào giữa (va='center') của chiều cao hàng động
        for (x_pos, text), key in zip(data_points, col_align.keys()):
            ax.text(x_pos, y_offset, text, fontsize=10, ha=col_align[key], va='center',
                    transform=ccrs.PlateCarree(), zorder=21)
        
        y_offset -= row_height # Di chuyển Y xuống bằng chiều cao động đã tính

    # --- BƯỚC 8: VẼ CHÚ THÍCH (DƯỚI CHÂN KHUNG, CÁCH ĐÁY 1.0) ---
    chuthich_path = os.path.join(icon_folder_path, 'chuthich.PNG')
    if os.path.exists(chuthich_path):
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox # Cần import
        legend_img = plt.imread(chuthich_path)
        oi_legend = OffsetImage(legend_img, zoom=0.2) 
        
        # Vẽ chú thích tại vị trí y_legend_center (đảm bảo đáy ảnh cách tọa độ 21.0 là 1.0)
        ab_legend = AnnotationBbox(oi_legend, (box_center_x, y_legend_center),
                                   xycoords='data', 
                                   box_alignment=(0.5, 0.5), # Căn tâm ảnh vào toạ độ
                                   frameon=False, zorder=22)
        ax.add_artist(ab_legend)
    else:
        print(f"CẢNH BÁO: Không tìm thấy tệp chú thích tại '{chuthich_path}'.")
# ==============================================================================
def main():
    try:
        df, time_window = load_and_preprocess_data(FILE_PATH, REQUIRED_COLS)

        map_extent = [93, 154, 3, 33]
        fig, ax = setup_map(map_extent)

        plot_wind_corridors(ax, df, time_window=time_window)
        plot_tracks(ax, df)
        plot_storm_track_and_icons(ax, df, ICON_PATHS)
        add_place_labels(ax)
        add_info_table(ax, df, ICON_FOLDER_PATH, time_window=time_window)

        plt.tight_layout(pad=0.5)
        
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plt.savefig(OUTPUT_FILE, dpi=1000)
        # plt.show() # Bỏ comment dòng này nếu bạn muốn xem ảnh hiển thị ngay sau khi chạy
        print(f"[OK] Đã lưu bản đồ mới tại: {OUTPUT_FILE}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Đã xảy ra lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()
