# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import io
import json
from math import radians, sin, cos, asin, sqrt
from folium.plugins import FloatImage

# Thư viện hình học chuyên sâu
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from cartopy import geodesic

# --- CẤU HÌNH ---
ICON_DIR = "icon"
DATA_FILE = "besttrack.xlsx"
CHUTHICH_IMG = os.path.join(ICON_DIR, "chuthich.PNG")
COL_R6, COL_R10, COL_RC = "#FFC0CB", "#FF6347", "#90EE90" 

st.set_page_config(page_title="Hệ thống Theo dõi Bão - Phong Le", layout="wide")

# --- 1. TIỆN ÍCH HÌNH HỌC (TẠO VÀNH KHĂN) ---
def get_geodesic_poly(lon, lat, radius_km):
    if radius_km <= 0: return None
    # Tạo đa giác vòng tròn chuẩn địa lý
    circle_points = geodesic.Geodesic().circle(lon=lon, lat=lat, radius=radius_km*1000, n_samples=100)
    return Polygon(circle_points)

def create_non_overlapping_swaths(dense_df):
    """Xử lý để các lớp không chồng lấn màu sắc"""
    polys_r6, polys_r10, polys_rc = [], [], []

    for _, row in dense_df.iterrows():
        p6 = get_geodesic_poly(row['lon'], row['lat'], row.get('r6', 0))
        p10 = get_geodesic_poly(row['lon'], row['lat'], row.get('r10', 0))
        pc = get_geodesic_poly(row['lon'], row['lat'], row.get('rc', 0))
        if p6: polys_r6.append(p6)
        if p10: polys_r10.append(p10)
        if pc: polys_rc.append(pc)

    # Hợp nhất các vòng tròn thành dải hành lang duy nhất
    union_r6 = unary_union(polys_r6) if polys_r6 else None
    union_r10 = unary_union(polys_r10) if polys_r10 else None
    union_rc = unary_union(polys_rc) if polys_rc else None

    # LOGIC TRỪ VÙNG: Lớp trên khoét lỗ lớp dưới
    final_rc = union_rc
    final_r10 = union_r10.difference(union_rc) if union_r10 and union_rc else union_r10
    
    # Hồng trừ đi tất cả vùng bên trong (Đỏ và Xanh)
    inner_all = unary_union([s for s in [union_r10, union_rc] if s is not None])
    final_r6 = union_r6.difference(inner_all) if union_r6 and inner_all else union_r6

    return final_r6, final_r10, final_rc

# --- CHƯƠNG TRÌNH CHÍNH ---
if os.path.exists(DATA_FILE):
    df = pd.read_excel(DATA_FILE)
    df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    
    from app_utils import densify_track # Giả sử bạn để hàm nội suy ở file utils
    dense_df = densify_track(df, step_km=10)

    m = folium.Map(location=[17.0, 115.0], zoom_start=5, tiles="OpenStreetMap")

    # Tạo các đa giác không chồng lấn
    f6, f10, fc = create_non_overlapping_swaths(dense_df)

    # Vẽ lên bản đồ dưới dạng GeoJson để kiểm soát màu sắc tuyệt đối
    for geom, color, opacity in [(f6, COL_R6, 0.5), (f10, COL_R10, 0.6), (fc, COL_RC, 0.7)]:
        if geom and not geom.is_empty:
            folium.GeoJson(
                mapping(geom),
                style_function=lambda x, c=color, o=opacity: {
                    'fillColor': c, 'color': c, 'weight': 1, 'fillOpacity': o
                }
            ).add_to(m)

    # Quỹ đạo & Icon
    folium.PolyLine(df[['lat', 'lon']].values.tolist(), color="black", weight=2).add_to(m)
    # ... (phần code Marker và Bảng tin giữ nguyên như cũ)

    st_folium(m, width="100%", height=750)
