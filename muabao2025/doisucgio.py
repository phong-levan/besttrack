# -*- coding: utf-8 -*-
"""
Đọc 'danhgia.xlsx' (sheet 'besttrack' + 'thang bão'),
tính cấp gió Beaufort từ 'wind (km/h)' và ghi vào cột 'cat'.
Xuất ra 'besttrack_capgio.xlsx' (sheet 'besttrack').
"""

import os
import re
import numpy as np
import pandas as pd

# ====== CẤU HÌNH ======
INPUT_XLSX  = "danhgia.xlsx"
OUTPUT_XLSX = "besttrack_capgio.xlsx"

# Aliases tên cột để chịu được khác biệt file
WIND_KMH_ALIASES = [
    "wind (km/h)", "wind(km/h)", "wind_km/h", "wind_kmh",
    "Gió (km/h)", "Gio (km/h)", "Gió (kmh)", "Gio (kmh)"
]

NUMERIC_COL_ALIASES = [
    # English headers trong ảnh
    "year", "mon", "day", "hour", "lat", "lon", "p (hPa)", "wind (kt)", "wind (km/h)",
    # Một số tiếng Việt có thể gặp
    "Năm", "Tháng", "Ngày", "Giờ", "Vĩ độ", "Dài.", "Áp suất (hPa)", "Gió (kt)", "Gió (km/h)"
]

def _coerce_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")

def _to_int_if_possible(series):
    """Chuyển về int nếu là số, nếu không để nguyên (dạng string)."""
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().all():
        return v.astype("Int64")
    return series.astype("string")

def find_first_existing(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None

def parse_cap_value(x):
    """Chuẩn hoá 'Cấp độ': nếu là 'B8'/'BF8' -> 8; nếu số -> số; nếu text -> giữ nguyên."""
    if pd.isna(x):
        return np.nan
    # Lấy chữ số đầu tiên xuất hiện
    m = re.search(r"\d+", str(x))
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    # Không phải số: trả lại nguyên văn
    return x

def map_beaufort_by_ranges(wind_kmh_series, df_ranges):
    """
    Ánh xạ km/h -> cấp gió theo khoảng [từ, đến] (bao gồm cả cận trên).
    Nếu 'đến' NaN: [từ, +∞).
    """
    rng = df_ranges.copy()

    # Kiểm tra cột trong 'thang bão'
    for col in ["từ", "đến"]:
        if col not in rng.columns:
            raise ValueError("Sheet 'thang bão' phải có cột 'từ' và 'đến'.")
    if "Cấp độ" not in rng.columns:
        raise ValueError("Sheet 'thang bão' phải có cột 'Cấp độ'.")

    rng["từ"]  = _coerce_numeric_series(rng["từ"])
    rng["đến"] = _coerce_numeric_series(rng["đến"])
    # Chuẩn hoá 'Cấp độ' về số nếu có ký tự
    rng["Cấp độ"] = rng["Cấp độ"].apply(parse_cap_value)

    # Sắp xếp để xử lý biên nhất quán
    rng = rng.sort_values(by=["từ", "đến"], na_position="last").reset_index(drop=True)

    lower  = rng["từ"].values
    upper  = rng["đến"].values
    labels = rng["Cấp độ"].values  # có thể là số hoặc chuỗi

    values = _coerce_numeric_series(wind_kmh_series).values
    out = np.full(len(values), np.nan, dtype=object)

    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if pd.isna(lo):
            continue
        lo = float(lo)
        if pd.isna(hi):
            mask = (values >= lo)                    # [lo, +∞)
        else:
            hi = float(hi)
            mask = (values >= lo) & (values <= hi)   # [lo, hi]  (bao gồm cận trên)
        out[mask] = labels[i]

    return pd.Series(out, index=wind_kmh_series.index, name="cat")  # ghi thẳng tên 'cat'

def clean_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ép số cho các cột số nếu có
    for c in NUMERIC_COL_ALIASES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="ignore")
    # 'cat' để dạng số nguyên nếu được, tránh NaN lẫn text
    if "cat" in out.columns:
        out["cat"] = _to_int_if_possible(out["cat"])
    # Thay NaN bằng rỗng để Excel không cảnh báo
    out = out.replace({np.nan: ""})
    return out

def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_XLSX}")

    besttrack = pd.read_excel(INPUT_XLSX, sheet_name="besttrack")
    thang_bao = pd.read_excel(INPUT_XLSX, sheet_name="thang bão")

    # Tìm cột gió km/h theo alias (ưu tiên cột tiếng Anh 'wind (km/h)')
    wind_col = find_first_existing(besttrack.columns, WIND_KMH_ALIASES)
    if wind_col is None:
        raise ValueError("Không tìm thấy cột tốc độ gió km/h (ví dụ 'wind (km/h)') trong sheet 'besttrack'.")

    # Tính 'cat' (cấp gió) theo thang bão
    cat_series = map_beaufort_by_ranges(besttrack[wind_col], thang_bao)

    # Ghi vào cột 'cat' (nếu đã có sẽ ghi đè; nếu chưa có thì thêm mới)
    besttrack["cat"] = cat_series

    # Làm sạch để Excel mở an toàn
    besttrack_clean = clean_for_excel(besttrack)

    # Xuất Excel bằng openpyxl (không cần cài thêm)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        besttrack_clean.to_excel(writer, sheet_name="besttrack", index=False)

    print(f"✅ Đã tạo file: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
