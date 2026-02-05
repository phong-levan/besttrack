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
OUTPUT_FILE = "bao_so_10_2025/2025092615Z.png"

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

    # <<< THAY ĐỔI: Cập nhật logic phân loại icon theo yêu cầu mới nhất
    def get_icon_name(row):
        wind_speed = row['cuong_do_bf']
        status = row['color_key']
        if pd.isna(wind_speed): return f"vungthap_{status}" # Mặc định là vùng thấp
        if wind_speed < 6:      return f"vungthap_{status}" # Dưới cấp 6: Vùng thấp
        if wind_speed < 8:      return f"atnd_{status}"     # Cấp 6, 7: Áp thấp nhiệt đới
        if wind_speed <= 11:    return f"bnd_{status}"      # Cấp 8 -> 11: Bão
        return f"sieubao_{status}" # Trên cấp 11: Siêu bão
        
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
    xticks = np.arange(100, 140, 5)
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
# Vùng gió
# ==============================================================================
def plot_wind_corridors(ax, df, time_window=None):
    sub = df[df['color_key'] != 'daqua'].copy()
    if time_window and 'dt' in sub.columns:
        start, end = time_window
        if start is not None and end is not None:
            sub = sub[(sub['dt'] >= start) & (sub['dt'] <= end)].copy()
    if sub.empty:
        print("[INFO] Không có điểm hiện tại/dự báo để vẽ vùng gió.")
        return

    lons = sub['lon'].astype(float).to_numpy()
    lats = sub['lat'].astype(float).to_numpy()
    r6   = pd.to_numeric(sub.get('ban_kinh_gio_6', np.nan), errors="coerce").fillna(0).to_numpy(float)
    r10  = pd.to_numeric(sub.get('ban_kinh_gio_10', np.nan), errors="coerce").fillna(0).to_numpy(float)
    rc   = pd.to_numeric(sub.get('ban_kinh_tam',   np.nan), errors="coerce").fillna(0).to_numpy(float)

    def _interp(a):
        return pd.Series(a).replace(0, np.nan).interpolate(limit_direction="both").fillna(0).to_numpy()
    r6, r10, rc = _interp(r6), _interp(r10), _interp(rc)

    if len(lons) >= 2:
        lons_d, lats_d, (r6_d, r10_d, rc_d) = densify_track(lons, lats, r6, r10, rc, step_km=25)
    else:
        lons_d, lats_d, r6_d, r10_d, rc_d = lons, lats, r6, r10, rc

    swath_r6  = union_of_circles(lons_d, lats_d, r6_d,  n_samples=420)
    swath_r10 = union_of_circles(lons_d, lats_d, r10_d, n_samples=420)
    swath_rc  = union_of_circles(lons_d, lats_d, rc_d,  n_samples=420)

    for g in iter_geoms(swath_r6):
        ax.add_geometries([g], crs=ccrs.PlateCarree(), facecolor=COL_R6,  edgecolor='none', alpha=0.50, zorder=2.5)
    for g in iter_geoms(swath_r10):
        ax.add_geometries([g], crs=ccrs.PlateCarree(), facecolor=COL_R10, edgecolor='none', alpha=0.60, zorder=2.8)
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

            # <<< THAY ĐỔI: Chỉnh kích thước icon vùng thấp
            # Nếu tên icon có chữ "vungthap", dùng độ phóng to nhỏ hơn
            if "vungthap" in icon_name:
                zoom_level = 0.08  # Kích thước nhỏ cho vùng thấp (bạn có thể chỉnh số này)
            else:
                zoom_level = 0.25  # Kích thước mặc định cho các icon khác

            oi = OffsetImage(img, zoom=zoom_level)
            # >>> KẾT THÚC THAY ĐỔI

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
        ax.text(lon, lat, name, fontsize=8, color='darkblue', weight='bold',
                ha='center', va='center', transform=ccrs.PlateCarree(), zorder=10,
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                              path_effects.Normal()])

# ==============================================================================
# Bảng thông tin
# ==============================================================================
def add_info_table(ax, df_forecast, icon_folder_path, time_window=None):
    box_lon_min, box_lon_max = 120, 135
    box_lat_min, box_lat_max = 21.5, 29.0

    sub = df_forecast[df_forecast['color_key'] != 'daqua'].copy()
    if time_window and 'dt' in sub.columns:
        start, end = time_window
        if start is not None and end is not None:
            sub = sub[(sub['dt'] >= start) & (sub['dt'] <= end)].copy()

    ax.add_patch(plt.Rectangle((box_lon_min, box_lat_min),
                               box_lon_max - box_lon_min, box_lat_max - box_lat_min,
                               facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.7,
                               transform=ccrs.PlateCarree(), zorder=20))
    
    box_center_x = box_lon_min + (box_lon_max - box_lon_min) / 2
    
    ax.text(box_center_x, box_lat_max - 0.8, "TIN BÃO KHẨN CẤP (Cơn bão số 10)",
            fontsize=12, weight='bold', ha='center', va='top', transform=ccrs.PlateCarree(), zorder=21)
    ax.text(box_center_x, box_lat_max - 1.8, "Tin phát lúc: 22 giờ 00 phút, 26/09/2025",
            fontsize=10, ha='center', va='top', transform=ccrs.PlateCarree(), zorder=21)

    col_pos = {
        "Ngay-gio": box_lon_min + 0.5,
        "Kinh do":  box_lon_min + 5.5,
        "Vi do":    box_lon_min + 7.8,
        "Cap gio":  box_lon_min + 10.3,
        "Pmin":     box_lon_min + 13.0,
    }
    col_align = {
        "Ngay-gio": 'left',
        "Kinh do": 'center',
        "Vi do": 'center',
        "Cap gio": 'center',
        "Pmin": 'center'
    }

    header_y = box_lat_max - 2.8
    headers = [("Ngay-gio", "Ngày - giờ"), ("Kinh do", "Kinh độ"),
               ("Vi do", "Vĩ độ"), ("Cap gio", "Cấp gió"), ("Pmin", "Pmin(mb)")]
    for key, text in headers:
        ax.text(col_pos[key], header_y, text, fontsize=10, weight='bold', ha=col_align[key], va='top',
                transform=ccrs.PlateCarree(), zorder=21)

    y_offset = box_lat_max - 3.6
    row_height = 0.7
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
        for (x_pos, text), key in zip(data_points, col_align.keys()):
            ax.text(x_pos, y_offset, text, fontsize=10, ha=col_align[key], va='top',
                    transform=ccrs.PlateCarree(), zorder=21)
        y_offset -= row_height

    chuthich_path = os.path.join(icon_folder_path, 'chuthich.PNG')
    if os.path.exists(chuthich_path):
        legend_img = plt.imread(chuthich_path)
        oi_legend = OffsetImage(legend_img, zoom=0.2)
        legend_y_pos = y_offset + row_height - 2
        ab_legend = AnnotationBbox(oi_legend, (box_center_x, legend_y_pos),
                                   xycoords='data', box_alignment=(0.5, 1.0),
                                   frameon=False, zorder=22)
        ax.add_artist(ab_legend)
    else:
        print(f"CẢNH BÁO: Không tìm thấy tệp chú thích tại '{chuthich_path}'.")

# ==============================================================================
# Main
# ==============================================================================
def main():
    try:
        df, time_window = load_and_preprocess_data(FILE_PATH, REQUIRED_COLS)

        map_extent = [97, 137, 3, 30]
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
        plt.show()
        print(f"[OK] Đã lưu: {OUTPUT_FILE}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Đã xảy ra lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()