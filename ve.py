# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import base64
import requests
import streamlit.components.v1 as components
from math import radians, sin, cos, asin, sqrt, pi
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH & DỮ LIỆU (NHÚNG TÀI KHOẢN TRỰC TIẾP)
# ==============================================================================
ICON_DIR = "icon"
FILE_OPT1 = "besttrack.csv"
FILE_OPT2 = "besttrack_capgio.xlsx"

# Nhúng Auth vào URL: http://user:pass@domain
# Ký tự @ trong mật khẩu được mã hóa thành %40
LINKS = {
    "CMA": "https://typhoon.nmc.cn/web.html", [cite: 5]
    "JMA": "https://www.jma.go.jp/bosai/map.html#5/13.582/115.84/&elem=root&typhoon=all&contents=typhoon&lang=en", [cite: 7]
    "RADAR": "http://hymetnet.gov.vn/radar/", [cite: 9]
    "WEATHER_OBS": "https://weatherobs.com/", [cite: 12]
    "GIO_VAN_HANH": "http://admin:ttdl%402021@222.255.11.82/Modules/Gio/MapWind.aspx", [cite: 15]
    "QUANTRAC_REALTIME": "http://admin:kttv%402021@tooldubao.tramthoitiet.vn/quantrac/kttv?province=KVBB", [cite: 13, 14]
    "GFS_MODEL": "https://www.tropicaltidbits.com/analysis/models/?model=gfs&region=ea&pkg=mslp_pcpn_frzn", [cite: 18]
    "KMA_POINT": "https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136", [cite: 20]
    "NCHMF_POINT": "http://rsfc:1234@swfdp-sea.com.vn/rfsc/" [cite: 21, 22]
}

st.set_page_config(page_title="Hệ thống giám sát", layout="wide")

# ==============================================================================
# 2. CSS GIAO DIỆN CỐ ĐỊNH
# ==============================================================================
st.markdown("""
    <style>
    .block-container { padding: 0 !important; }
    header, footer { display: none !important; }
    iframe { width: 100% !important; height: 95vh !important; border: none !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. MAIN APP
# ==============================================================================
def main():
    with st.sidebar:
        st.title("Dữ liệu khí tượng")
        topic = st.radio("CHỌN CHẾ ĐỘ:", 
                        ["Bản đồ Bão", "Quan trắc thời gian thực", "Dữ liệu quan trắc", "Dự báo thời tiết & khí hậu"])
        st.markdown("---")

        active_link = None
        
        if topic == "Bản đồ Bão":
            storm_opt = st.selectbox("Nguồn dự báo:", 
                                    ["Dự báo cá nhân (Besttrack)", "Dự báo của CMA", "Dự báo của JMA"]) [cite: 3, 4, 6]
            if "CMA" in storm_opt: active_link = LINKS["CMA"]
            elif "JMA" in storm_opt: active_link = LINKS["JMA"]

        elif topic == "Quan trắc thời gian thực":
            obs_real = st.radio("Loại dữ liệu:", ["Số liệu radar", "Số liệu vệ tinh"]) [cite: 8, 9, 10]
            if "radar" in obs_real: active_link = LINKS["RADAR"]

        elif topic == "Dữ liệu quan trắc":
            obs_src = st.radio("Nguồn dữ liệu:", 
                              ["Bản đồ gió (Vận hành)", "Thời tiết (WeatherObs)", "Quan trắc thời gian thực (Kttv)"]) [cite: 11, 12, 13, 15]
            if "gió" in obs_src.lower(): active_link = LINKS["GIO_VAN_HANH"]
            elif "weatherobs" in obs_src.lower(): active_link = LINKS["WEATHER_OBS"]
            else: active_link = LINKS["QUANTRAC_REALTIME"]

        elif topic == "Dự báo thời tiết & khí hậu":
            fore_src = st.radio("Mô hình:", ["Dự báo mô hình (GFS)", "Dự báo điểm (KMA)", "Dự báo điểm (NCHMF)"]) [cite: 16, 17, 19, 21]
            if "GFS" in fore_src: active_link = LINKS["GFS_MODEL"]
            elif "KMA" in fore_src: active_link = LINKS["KMA_POINT"]
            else: active_link = LINKS["NCHMF_POINT"]

    # --- HIỂN THỊ NỘI DUNG ---
    if active_link:
        # Hỗ trợ nút mở tab mới nếu Iframe bị trình duyệt chặn Auth
        if "@" in active_link:
            st.link_button("🌐 Truy cập trực tiếp hệ thống", active_link)
        components.iframe(active_link, scrolling=True)
    else:
        # Mặc định hiển thị bản đồ Folium cho Besttrack/Vệ tinh
        st.info("Hệ thống đang sẵn sàng. Vui lòng chọn nguồn dữ liệu từ Sidebar.")

if __name__ == "__main__":
    main()
