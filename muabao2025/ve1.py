# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from cartopy.io import shapereader as shpreader
from cartopy.feature import ShapelyFeature

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*longdouble.*Signature.*", category=UserWarning)

# ==============================================================================
# Cấu hình
# ==============================================================================
FILE_PATH        = "besttrack_capgio.xlsx"
ICON_FOLDER_PATH = "/home/lephong/besttrack/icon"
OUTPUT_FILE      = "mua_bao_2025_new.png"
SHP_PATH         = "/home/lephong/ve_excel/vn/RG_Tinh_3857.shp"     # <== đường dẫn shapefile (cần .shp, .shx, .dbf)

SEASON_MODE     = True
YEAR_FILTER     = 2025
TRACK_COLOR     = "black"
USE_NAME_LABEL  = True

# Cột trong Excel tiếng Việt -> tên chuẩn nội bộ
REQUIRED_COLS = {
    "Thời điểm": "thoi_diem",
    "tên bão": "name",
    "năm": "year",
    "tháng": "mon",
    "ngày": "day",
    "giờ": "hour",
    "vĩ độ": "lat",
    "kinh độ": "lon",
    "khí áp (hPa)": "pmin",
    "gió (kt)": "wind_kt",
    "gió (km/h)": "wind_kmh",
    "cấp bão": "cuong_do_bf",
    "Ngày - giờ": "ngay_gio",
}

# tên icon cơ bản (không kèm đuôi); loader sẽ tự thử .png/.PNG/.jpg/.jpeg
ICON_BASES = {
    "vungthap_daqua": "vungthapdaqua",
    "atnd_daqua":     "atnddaqua",
    "bnd_daqua":      "bnddaqua",
    "sieubao_daqua":  "sieubaodaqua",
    "vungthap_dubao": "vungthapdubao",
    "atnd_dubao":     "atnd",
    "bnd_dubao":      "bnd",
    "sieubao_dubao":  "sieubao",
}

# ==============================================================================
# Tiện ích chung
# ==============================================================================
def to_datetime_series(df):
    if all(k in df.columns for k in ["year","mon","day","hour"]):
        return pd.to_datetime(dict(
            year = pd.to_numeric(df["year"], errors="coerce"),
            month= pd.to_numeric(df["mon"],  errors="coerce"),
            day  = pd.to_numeric(df["day"],  errors="coerce"),
            hour = pd.to_numeric(df["hour"], errors="coerce").fillna(0)
        ), errors="coerce")
    return pd.to_datetime(pd.Series([None]*len(df)))

def get_time_window(*args, **kwargs):
    return None, None

def find_icon_path(base):
    for ext in (".png",".PNG",".jpg",".JPG",".jpeg",".JPEG"):
        p = os.path.join(ICON_FOLDER_PATH, base + ext)
        if os.path.exists(p):
            return p
    return None

def build_icon_cache():
    cache = {}
    for key, base in ICON_BASES.items():
        p = find_icon_path(base)
        if p:
            cache[key] = plt.imread(p)
        else:
            print(f"[ICON] Không tìm thấy file cho '{key}' (gốc: {base}.*) trong '{ICON_FOLDER_PATH}/'")
    return cache

# ==============================================================================
# Đọc & tiền xử lý  (đúng block thầy yêu cầu, đã sửa lỗi dấu '}')
# ==============================================================================
def load_and_preprocess_data(file_path, required_cols):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp dữ liệu tại {file_path}")

    sheets = pd.read_excel(file_path, sheet_name=None)

    all_dfs = []
    for name, sdf in sheets.items():
        cols = [c.lower().strip() for c in sdf.columns]
        if "lat" in cols and "lon" in cols or "vĩ độ" in cols and "kinh độ" in cols:
            all_dfs.append(sdf)

    if not all_dfs:
        raise ValueError("Không tìm thấy sheet nào chứa cột 'lat/lon' (hoặc 'vĩ độ/kinh độ').")

    df_excel = pd.concat(all_dfs, ignore_index=True)

    # ---- SỬA LỖI: đóng dấu '}' cho dict-comprehension ----
    rename_map = {k: v for k, v in required_cols.items() if k in df_excel.columns}
    df = df_excel.rename(columns=rename_map)

    for original_col, new_col in required_cols.items():
        if original_col not in df_excel.columns and new_col not in df.columns:
            print(f"CẢNH BÁO: Thiếu cột '{original_col}' → tạo '{new_col}'=NaN.")
            df[new_col] = np.nan

    df = df.dropna(subset=["lat", "lon"]).copy()
    if 'ngay_gio' in df.columns and df['ngay_gio'].notna().any():
        df["dt"] = pd.to_datetime(df["ngay_gio"], dayfirst=True, errors="coerce")
    else:
        df["dt"] = to_datetime_series(df)
    df = df.dropna(subset=['dt'])

    if YEAR_FILTER is not None and "year" in df.columns:
        df = df[pd.to_numeric(df["year"], errors="coerce") == YEAR_FILTER]

    if 'thoi_diem' in df.columns:
        df["color_key"] = df["thoi_diem"].astype(str).str.lower().apply(
            lambda x: 'daqua' if "quá khứ" in x or "qua khu" in x else 'dubao'
        )
    else:
        df["color_key"] = 'daqua'

    def get_icon_name(row):
        wind_speed = pd.to_numeric(row.get('cuong_do_bf', np.nan), errors="coerce")
        status = row.get('color_key', 'daqua')
        if pd.isna(wind_speed): return f"vungthap_{status}"
        if wind_speed < 6:      return f"vungthap_{status}"
        if wind_speed < 8:      return f"atnd_{status}"
        if wind_speed <= 11:    return f"bnd_{status}"
        return f"sieubao_{status}"

    df['icon_name'] = df.apply(get_icon_name, axis=1)
    start, end = get_time_window(sheets, df)
    return df.sort_values(["name","dt"]).reset_index(drop=True), (start, end)

# ==============================================================================
# Bản đồ & shapefile
# ==============================================================================
def setup_map(extent):
    fig = plt.figure(figsize=(12, 9), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#d6efff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", edgecolor='black', linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3, linestyle="--", edgecolor='gray', zorder=1.4)
    xticks = np.arange(100, 140, 5); yticks = np.arange(5, 35, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()); ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.gridlines(xlocs=xticks, ylocs=yticks, linewidth=0.5, color="k", alpha=0.35, linestyle="--", draw_labels=False)
    return fig, ax

def add_place_labels(ax):
    places = {
        "Hà Nội": (105.8, 21.0), "Đà Nẵng": (108.2, 16.0), "TP.HCM": (106.7, 10.8),
        "DK.Hoàng Sa": (111.7, 16.0), "DK.Trường Sa": (113.8, 8.5)
    }
    for name, (lon, lat) in places.items():
        ax.text(lon, lat, name, fontsize=8, color='darkblue', weight='bold',
                ha='center', va='center', transform=ccrs.PlateCarree(), zorder=10,
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                              path_effects.Normal()])

def add_shapefile(ax, shp_path):
    """Vẽ ranh giới tỉnh từ shapefile (WGS84)."""
    if not os.path.exists(shp_path):
        print(f"[SHP] Không tìm thấy shapefile: {shp_path}")
        return
    try:
        reader = shpreader.Reader(shp_path)
        feat = ShapelyFeature(reader.geometries(), ccrs.PlateCarree())
        # viền đậm hơn đường biên quốc gia một chút
        ax.add_feature(feat, facecolor='none', edgecolor='#444', linewidth=0.6, zorder=1.6)
    except Exception as e:
        print(f"[SHP] Lỗi đọc shapefile: {e}")

# ==============================================================================
# Vẽ quỹ đạo, icon, nhãn
# ==============================================================================
def plot_tracks_black(ax, df, lw=1.8):
    if df.empty: return
    df = df.sort_values('dt')
    ax.plot(df['lon'], df['lat'], color=TRACK_COLOR, linewidth=lw, zorder=6, transform=ccrs.PlateCarree())

def plot_icons(ax, df, image_cache, zoom_low=0.10, zoom_storm=0.32):
    for _, row in df.iterrows():
        key = row.get("icon_name")
        if not key or key not in image_cache: 
            continue
        img = image_cache[key]
        zoom = zoom_low if "vungthap" in key else zoom_storm
        oi = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(oi, (float(row["lon"]), float(row["lat"])),
                            frameon=False, xycoords='data', zorder=12,
                            box_alignment=(0.5,0.5))
        ax.add_artist(ab)

def label_storm_at_genesis(ax, sub):
    if sub.empty: return
    first = sub.sort_values('dt').iloc[0]
    ax.text(float(first['lon']), float(first['lat']), f" {first['name']}",
            color="black", fontsize=9, weight='bold',
            ha='left', va='center', transform=ccrs.PlateCarree(), zorder=15,
            path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                          path_effects.Normal()])

# ==============================================================================
# Main
# ==============================================================================
def main():
    try:
        df, _ = load_and_preprocess_data(FILE_PATH, REQUIRED_COLS)

        # Extent tự động
        if not df.empty:
            lat_min, lat_max = float(df['lat'].min()), float(df['lat'].max())
            lon_min, lon_max = float(df['lon'].min()), float(df['lon'].max())
            pad_lat = max(1.5, (lat_max - lat_min) * 0.15)
            pad_lon = max(1.5, (lon_max - lon_min) * 0.15)
            extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]
        else:
            extent = [97, 137, 3, 30]

        fig, ax = setup_map(extent)
        add_shapefile(ax, SHP_PATH)   # <== vẽ ranh giới 34 tỉnh
        add_place_labels(ax)

        image_cache = build_icon_cache()

        if SEASON_MODE and "name" in df.columns:
            for storm in df['name'].dropna().unique():
                sub = df[df['name'] == storm].copy()
                plot_tracks_black(ax, sub, lw=1.8)
                plot_icons(ax, sub, image_cache, zoom_low=0.10, zoom_storm=0.32)
                if USE_NAME_LABEL:
                    label_storm_at_genesis(ax, sub)
            ax.set_title(f"MÙA BÃO {YEAR_FILTER} – Quỹ đạo best track (đường đen, có icon, overlay VN 34 tỉnh)",
                         fontsize=13, weight='bold')
        else:
            plot_tracks_black(ax, df, lw=1.8)
            plot_icons(ax, df, image_cache, zoom_low=0.10, zoom_storm=0.32)

        plt.tight_layout(pad=0.5)
        out_dir = os.path.dirname(OUTPUT_FILE)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        plt.savefig(OUTPUT_FILE, dpi=800)
        print(f"[OK] Đã lưu bản đồ: {OUTPUT_FILE}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Đã xảy ra lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()
