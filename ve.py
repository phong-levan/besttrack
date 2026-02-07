import os
import pandas as pd
import numpy as np

# Giả sử đường dẫn thư mục icon (bạn có thể thay đổi đường dẫn này)
ICON_FOLDER_PATH = "icon"

# 1. Định nghĩa Dictionary đường dẫn Icon
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

# 2. Hàm xử lý dữ liệu
def load_and_preprocess_data(file_path, required_cols):
    # Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp dữ liệu tại {file_path}")

    # Đọc tất cả các sheet
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        raise ValueError(f"Không thể đọc file Excel: {e}")
    
    all_dfs = []
    # Lọc các sheet có chứa cột lat/lon
    for name, sdf in sheets.items():
        # Chuẩn hóa tên cột trong sheet về chữ thường để kiểm tra
        cols = [str(c).lower().strip() for c in sdf.columns]
        if "lat" in cols and "lon" in cols:
            all_dfs.append(sdf)
    
    if not all_dfs:
         raise ValueError("Không tìm thấy sheet nào chứa cột 'lat' và 'lon' trong file Excel.")
    
    # Gộp dữ liệu
    df_excel = pd.concat(all_dfs, ignore_index=True)

    # Đổi tên cột theo required_cols (Ví dụ: 'Cường độ' -> 'cuong_do_bf', 'Thời điểm' -> 'thoi_diem')
    # Lưu ý: Cần đảm bảo keys trong required_cols khớp với tên cột trong Excel
    df = df_excel.rename(columns={k: v for k, v in required_cols.items() if k in df_excel.columns})

    # Tạo các cột thiếu nếu cần thiết
    for original_col, new_col in required_cols.items():
        if new_col not in df.columns:
            # print(f"CẢNH BÁO: Thiếu cột '{original_col}' (mapped to '{new_col}') → tạo NaN.")
            df[new_col] = np.nan

    # Loại bỏ dòng không có tọa độ
    df = df.dropna(subset=["lat", "lon"]).copy()

    # Xử lý ngày giờ (Giả định hàm to_datetime_series đã được định nghĩa ở nơi khác)
    if 'Ngày - giờ' in df_excel.columns:
        df['ngay_gio'] = df_excel['Ngày - giờ'].astype(str)
    
    # Gọi hàm xử lý thời gian (Bạn cần đảm bảo hàm này đã có trong code của bạn)
    # Nếu chưa có, hãy dùng: df["dt"] = pd.to_datetime(df["datetime_str_col"], dayfirst=True, errors='coerce')
    try:
        df["dt"] = to_datetime_series(df) 
        df = df.dropna(subset=['dt'])
    except NameError:
        print("Lỗi: Hàm 'to_datetime_series' chưa được định nghĩa.")
        return pd.DataFrame(), (None, None)

    # Xác định trạng thái quá khứ/dự báo
    # Logic: Nếu cột 'thoi_diem' chứa chữ "quá khứ" (case-insensitive) -> 'daqua', ngược lại -> 'dubao'
    if 'thoi_diem' in df.columns:
        df["status_suffix"] = df["thoi_diem"].astype(str).apply(
            lambda x: 'daqua' if "quá khứ" in x.lower() else 'dubao'
        )
    else:
        df["status_suffix"] = 'dubao'

    # Xử lý cột cường độ gió để so sánh số học
    if 'cuong_do_bf' in df.columns:
        df['cuong_do_bf'] = pd.to_numeric(df['cuong_do_bf'], errors='coerce')

    # Hàm nội bộ xác định tên icon dựa trên cấp gió và trạng thái
    def get_icon_name(row):
        wind_speed = row.get('cuong_do_bf', np.nan)
        status = row.get('status_suffix', 'dubao')
        
        # Logic phân loại bão
        if pd.isna(wind_speed): 
            prefix = "vungthap"
        elif wind_speed < 6:
            prefix = "vungthap"
        elif wind_speed < 8: # Cấp 6-7
            prefix = "atnd"
        elif wind_speed <= 11: # Cấp 8-11
            prefix = "bnd"
        else: # Cấp >= 12
            prefix = "sieubao"
            
        return f"{prefix}_{status}"
        
    df['icon_name'] = df.apply(get_icon_name, axis=1)

    # Lấy khoảng thời gian (Giả định hàm get_time_window đã được định nghĩa)
    try:
        start, end = get_time_window(sheets, df)
    except NameError:
        start, end = df['dt'].min(), df['dt'].max()

    return df.sort_values("dt").reset_index(drop=True), (start, end)
