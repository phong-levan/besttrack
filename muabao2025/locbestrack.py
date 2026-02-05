import pandas as pd
import numpy as np
import os

def xu_ly_du_lieu_bao(input_file, output_file):
    print(f"Đang xử lý file: {input_file}")
    
    if not os.path.exists(input_file):
        print("LỖI: Không tìm thấy file đầu vào.")
        return

    # Đọc file (hỗ trợ cả xlsx và csv)
    try:
        if input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file, engine='openpyxl')
        else:
            df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"Lỗi mở file: {e}")
        return

    # 1. Xử lý thời gian và lọc lỗi
    if 'ISO_TIME' in df.columns:
        df['ISO_TIME'] = df['ISO_TIME'].astype(str) # Chuyển thành chuỗi trước
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df = df.dropna(subset=['ISO_TIME'])
    
    # 2. Chuyển đổi số liệu
    for col in ['LAT', 'LON', 'USA_WIND', 'USA_PRES']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Lọc Biển Đông (100-120E, 5-25N)
    mask = (df['LON'] >= 100) & (df['LON'] <= 120) & \
           (df['LAT'] >= 5) & (df['LAT'] <= 25)
    ids = df.loc[mask, 'SID'].unique()
    
    print(f"-> Tìm thấy {len(ids)} cơn bão/ATNĐ hoạt động trong Biển Đông.")

    # 4. Lấy dữ liệu trọn vòng đời
    df_filtered = df[df['SID'].isin(ids)].copy()
    
    # --- QUAN TRỌNG: RESET INDEX để tránh lỗi dòng trống ---
    df_filtered = df_filtered.reset_index(drop=True)
    # -----------------------------------------------------

    # 5. Tính cấp bão
    def tinh_cap(w):
        if pd.isna(w) or w < 1: return 0
        if w <= 33: return 7 # < Cấp 8
        if w <= 40: return 8
        if w <= 47: return 9
        if w <= 55: return 10
        if w <= 63: return 11
        if w <= 71: return 12
        if w <= 80: return 13
        if w <= 89: return 14
        if w <= 99: return 15
        if w <= 109: return 16
        return 17

    # 6. Tạo bảng kết quả
    out = pd.DataFrame()
    out['Thời điểm'] = ['quá khứ'] * len(df_filtered)
    out['tên bão'] = df_filtered['NAME']
    out['năm'] = df_filtered['ISO_TIME'].dt.year
    out['tháng'] = df_filtered['ISO_TIME'].dt.month
    out['ngày'] = df_filtered['ISO_TIME'].dt.day
    out['giờ'] = df_filtered['ISO_TIME'].dt.hour
    out['vĩ độ'] = df_filtered['LAT']
    out['kinh độ'] = df_filtered['LON']
    out['khí áp (hPa)'] = df_filtered.get('USA_PRES', 0)
    out['gió (kt)'] = df_filtered.get('USA_WIND', 0).fillna(0)
    out['gió (km/h)'] = (out['gió (kt)'] * 1.852).round(1)
    out['cấp bão'] = out['gió (kt)'].apply(tinh_cap)

    # Sắp xếp
    out = out.sort_values(by=['tên bão', 'năm', 'tháng', 'ngày', 'giờ'])
    
    # Xuất file
    out.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"XONG! Kết quả lưu tại: {output_file}")

if __name__ == "__main__":
    # Nhớ sửa tên file input cho đúng với máy bạn
    xu_ly_du_lieu_bao('besttrack_2025.xlsx', 'ket_qua_day_du.csv')