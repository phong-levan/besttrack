# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Hình học để hợp nhất vùng gió (giữ nguyên, nếu không có bán kính thì phần này tự bỏ qua)
from cartopy import geodesic
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*longdouble.*Signature.*", category=UserWarning)

# ==============================================================================
# Cấu hình
# ==============================================================================
FILE_PATH = "besttrack_capgio.xlsx"   # <= đổi đúng tên file của thầy
ICON_FOLDER_PATH = 'icon'
OUTPUT_FILE = "mua_bao_2025_newwa.png"

# Bật chế độ mùa bão và năm cần lọc
SEASON_MODE   = True
YEAR_FILTER   = 2025            # lọc theo cột 'năm'
USE_NAME_LABEL = True           # hiện nhãn tên bão ở cuối quỹ đạo

# Màu vùng gió + đường đi
COL_R6   = "#FFC0CB"
COL_R10  = "#FF6347"
COL_RC   = "#90EE90"
COL_TRK_PAST = "black"
COL_TRK_CURF = "black"

# Bảng màu phân biệt nhiều bão trong mùa
COLOR_CYCLE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
]

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
# Chuẩn hoá cột tiếng Việt -> tên chuẩn nội bộ
# ==============================================================================
def normalize_columns_vn(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "thời điểm":      mapping[c] = "thoi_diem"
        elif cl == "tên bão":      mapping[c] = "name"
        elif cl == "năm":          mapping[c] = "year"
        elif cl == "tháng":        mapping[c] = "mon"
        elif cl == "ngày":         mapping[c] = "day"
        elif cl == "giờ":          mapping[c] = "hour"
        elif cl == "vĩ độ":        mapping[c] = "lat"
        elif cl == "kinh độ":      mapping[c] = "lon"
        elif cl in ("khí áp (hpa)","khí áp (hpa) "): mapping[c] = "pmin"
        elif cl == "gió (kt)":     mapping[c] = "wind_kt"
        elif cl == "gió (km/h)":   mapping[c] = "wind_kmh"
        elif cl == "cấp bão":      mapping[c] = "cuong_do_bf"  # Beaufort
        else:
            mapping[c] = c  # giữ nguyên nếu không khớp
    out = df.rename(columns=mapping)
    return out

# ==============================================================================
# Thời gian
# ==============================================================================
def to_datetime_series(df):
    """
    Tạo Series thời gian 'dt' (naive) từ các cột year/mon/day/hour đã chuẩn hoá.
    Nếu thiếu, trả về NaT cho hàng đó.
    """
    need = ["year","mon","day","hour"]
    if all(k in df.columns for k in need):
        s = pd.to_datetime(dict(
            year=pd.to_numeric(df["year"], errors='coerce'),
            month=pd.to_numeric(df["mon"],  errors='coerce'),
            day=pd.to_numeric(df["day"],   errors='coerce'),
            hour=pd.to_numeric(df["hour"], errors='coerce').fillna(0)
        ), errors="coerce")
        return s
    return pd.to_datetime(pd.Series([None]*len(df)))

def get_time_window(*args, **kwargs):
    # Không dùng time window trong chế độ mùa bão
    return None, None

# ==============================================================================
# Đọc & tiền xử lý
# ==============================================================================
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp dữ liệu tại {file_path}")

    # Đọc sheet 'besttrack' (đúng như file của thầy)
    dfbt = pd.read_excel(file_path, sheet_name="besttrack")
    dfbt = normalize_columns_vn(dfbt)

    # Kiểm tra cột bắt buộc
    for k in ["name","year","mon","day","hour","lat","lon"]:
        if k not in dfbt.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {k}")

    # Tạo dt
    dfbt["dt"] = to_datetime_series(dfbt)
    dfbt = dfbt.dropna(subset=["dt","lat","lon"]).copy()

    # Đổi kiểu số cơ bản
    for c in ["lat","lon","pmin","wind_kt","wind_kmh","cuong_do_bf","year","mon","day","hour"]:
        if c in dfbt.columns:
            dfbt[c] = pd.to_numeric(dfbt[c], errors="coerce")

    # Màu/loại thời điểm: 'quá khứ' -> daqua, khác -> dubao
    if "thoi_diem" in dfbt.columns:
        dfbt["color_key"] = dfbt["thoi_diem"].astype(str).str.lower().apply(
            lambda x: "daqua" if "quá khứ" in x or "qua khu" in x else "dubao"
        )
    else:
        dfbt["color_key"] = "daqua"

    # Bán kính (không có trong file -> để NaN, phần hành lang gió sẽ bỏ qua)
    for c in ["ban_kinh_gio_6","ban_kinh_gio_10","ban_kinh_tam"]:
        if c not in dfbt.columns:
            dfbt[c] = np.nan

    # Icon theo cấp Beaufort
    def get_icon_name(row):
        bf = row.get("cuong_do_bf", np.nan)
        status = row.get("color_key","daqua")
        if pd.isna(bf): return f"vungthap_{status}"
        if bf < 6:      return f"vungthap_{status}"
        if bf < 8:      return f"atnd_{status}"
        if bf <= 11:    return f"bnd_{status}"
        return f"sieubao_{status}"
    dfbt["icon_name"] = dfbt.apply(get_icon_name, axis=1)

    # Lọc theo năm
    if YEAR_FILTER is not None and "year" in dfbt.columns:
        dfbt = dfbt[dfbt["year"] == YEAR_FILTER]

    # Chuỗi hiển thị ngày-giờ (nếu cần dùng)
    dfbt["ngay_gio"] = dfbt["dt"].dt.strftime("%d/%m/%Y %H:%M")

    return dfbt.sort_values(["name","dt"]).reset_index(drop=True), (None,None)

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
# Hình học (hành lang gió – sẽ tự bỏ qua nếu không có bán kính)
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

def plot_wind_corridors(ax, df, time_window=None):
    sub = df[df['color_key'] != 'daqua'].copy()
    if time_window and 'dt' in sub.columns:
        start, end = time_window
        if start is not None and end is not None:
            sub = sub[(sub['dt'] >= start) & (sub['dt'] <= end)].copy()
    if sub.empty:
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

    swath_r6  = union_of_circles(lons_d, lats_d, r6_d,  n_samples=360)
    swath_r10 = union_of_circles(lons_d, lats_d, r10_d, n_samples=360)
    swath_rc  = union_of_circles(lons_d, lats_d, rc_d,  n_samples=360)

    plot_swath_r10 = None
    if swath_r10:
        plot_swath_r10 = swath_r10.difference(swath_rc) if swath_rc else swath_r10

    plot_swath_r6 = None
    if swath_r6:
        inner = unary_union([s for s in [swath_r10, swath_rc] if s is not None and not s.is_empty])
        plot_swath_r6 = swath_r6.difference(inner) if inner else swath_r6

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
# Đường đi & Icon
# ==============================================================================
def plot_tracks(ax, df, color_past=None, color_curf=None, lw=1.6):
    cp = color_past or COL_TRK_PAST
    cc = color_curf or COL_TRK_CURF

    df_past = df[df['color_key'] == 'daqua'].copy().sort_values('dt')
    df_curf = df[df['color_key'] != 'daqua'].copy().sort_values('dt')

    if not df_past.empty:
        ax.plot(df_past['lon'], df_past['lat'], color=cp, linewidth=lw, zorder=5, transform=ccrs.PlateCarree())
    if not df_curf.empty:
        ax.plot(df_curf['lon'], df_curf['lat'], color=cc, linewidth=lw, linestyle="-", zorder=6, transform=ccrs.PlateCarree())
    if not df_past.empty and not df_curf.empty:
        ax.plot([df_past['lon'].iloc[-1], df_curf['lon'].iloc[0]],
                [df_past['lat'].iloc[-1], df_curf['lat'].iloc[0]],
                color=cc, linewidth=lw, linestyle="-", zorder=5.5, transform=ccrs.PlateCarree())

def plot_storm_track_and_icons(ax, df, icon_paths, zoom_low=0.08, zoom_storm=0.25):
    image_cache = {}
    for name, path in icon_paths.items():
        if os.path.exists(path):
            image_cache[name] = plt.imread(path)

    for _, row in df.iterrows():
        icon_name = row.get('icon_name', None)
        if not icon_name or icon_name not in image_cache:
            continue
        img = image_cache[icon_name]
        zoom_level = zoom_low if "vungthap" in icon_name else zoom_storm
        oi = OffsetImage(img, zoom=zoom_level)
        ab = AnnotationBbox(oi, (row["lon"], row["lat"]), frameon=False,
                            xycoords='data', zorder=12, box_alignment=(0.5,0.5))
        ax.add_artist(ab)

# ==============================================================================
# Nhãn địa lý & nhãn tên bão
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

def label_storm(ax, sub, label, color="k"):
    if sub.empty: return
    last = sub.sort_values('dt').iloc[-1]
    ax.text(float(last['lon']), float(last['lat']), f" {label}",
            color=color, fontsize=9, weight='bold',
            transform=ccrs.PlateCarree(), zorder=15,
            ha='left', va='center',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                          path_effects.Normal()])

# ==============================================================================
# Main
# ==============================================================================
def main():
    try:
        df, _ = load_and_preprocess_data(FILE_PATH)

        # Tự xác định extent theo dữ liệu
        if not df.empty:
            lat_min, lat_max = float(df['lat'].min()), float(df['lat'].max())
            lon_min, lon_max = float(df['lon'].min()), float(df['lon'].max())
            pad_lat = max(1.5, (lat_max - lat_min) * 0.15)
            pad_lon = max(1.5, (lon_max - lon_min) * 0.15)
            extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]
        else:
            extent = [97, 137, 3, 30]

        fig, ax = setup_map(extent)
        add_place_labels(ax)

        if SEASON_MODE and 'name' in df.columns:
            legend_handles = []
            names = [str(x) for x in df['name'].dropna().unique()]
            nstorm = len(names)
            colors = (COLOR_CYCLE * ((nstorm//len(COLOR_CYCLE))+1))[:nstorm]

            for color, name in zip(colors, names):
                sub = df[df['name'] == name].copy()
                plot_tracks(ax, sub, color_past=color, color_curf=color, lw=1.8)
                plot_storm_track_and_icons(ax, sub, ICON_PATHS)
                if USE_NAME_LABEL:
                    label_storm(ax, sub, name, color=color)
                legend_handles.append(mlines.Line2D([], [], color=color, lw=2, label=name))

            ax.set_title(f"MÙA BÃO TRÊN BIỂN ĐÔNG NĂM {YEAR_FILTER}", fontsize=13, weight='bold')
            if legend_handles:
                ax.legend(handles=legend_handles, loc='lower left', fontsize=8, frameon=True)

        else:
            plot_tracks(ax, df)
            plot_storm_track_and_icons(ax, df, ICON_PATHS)

        plt.tight_layout(pad=0.5)
        out_dir = os.path.dirname(OUTPUT_FILE)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        plt.savefig(OUTPUT_FILE, dpi=800)
        print(f"[OK] Đã lưu bản đồ: {OUTPUT_FILE}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Đã xảy ra lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()
