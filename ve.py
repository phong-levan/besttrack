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
# 1. CẤU HÌNH ĐƯỜNG DẪN (ĐÃ FIX SẠCH LỖI SYNTAX & CẬP NHẬT THỜI GIAN THỰC)
# ==============================================================================
LINKS = {
    "CMA": "https://typhoon.nmc.cn/web.html",
    "JMA": "https://www.jma.go.jp/bosai/map.html#5/13.582/115.84/&elem=root&typhoon=all&contents=typhoon&lang=en",
    "RADAR": "http://hymetnet.gov.vn/radar/",
    "WEATHER_OBS": "https://weatherobs.com/",
    # Tự động đăng nhập cho Bản đồ gió (admin:ttdl@2021)
    "GIO_VAN_HANH": "http://admin:ttdl%402021@222.255.11.82/Modules/Gio/MapWind.aspx",
    # Quan trắc thời gian thực (admin:kttv@2021)
    "QUANTRAC_REALTIME": "http://admin:kttv%402021@tooldubao.tramthoitiet.vn/quantrac/kttv?province=KVBB&day=09%2F02%2F2026",
    # Dự báo GFS cập nhật thời gian thực (09/02/2026)
    "GFS_MODEL": "https://www.tropicaltidbits.com/analysis/models/?model=gfs&region=ea&pkg=mslp_pcpn_frzn&runtime=2026020900&fh=6",
    # Dự báo điểm KMA cập nhật thời gian thực (09/02/2026)
    "KMA_POINT": "https://www.kma.go.kr/ema/nema03_kim/rall/detail.jsp?opt1=epsgram&opt2=VietNam&opt3=136&tm=2026.02.09.00&delta=000&ftm=2026.02.09.00",
    # Dự báo điểm NCHMF (rsfc:1234)
    "NCHMF_POINT": "http://rsfc:1234@swfdp-sea.com.vn/rfsc/"
}

st.set_page_config(page_title="Hệ thống giám sát KTTV", layout="wide")

# ==============================================================================
# 2. CSS GIAO DIỆN CỐ ĐỊNH
# ==============================================================================
st.markdown("""
    <style>
    .block-container { padding: 0 !important; }
    header, footer { display: none !important; }
    iframe { width: 100% !important; height: 95vh !important; border: none !important; }
    section[data-testid="stSidebar"] { width: 320px !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. APP CHÍNH
# ==============================================================================
def main():
    with st.sidebar:
        st.title("Giám Sát Khí Tượng")
        topic = st.radio("CHỌN CHẾ ĐỘ:", 
                        ["Bản đồ Bão", "Quan trắc thời gian thực", "Dữ liệu quan trắc", "Dự báo thời tiết & khí hậu"])
        st.markdown("---")

        active_link = None
        
        if topic == "Bản đồ Bão":
            # [cite: 1, 2, 3, 4, 6]
            storm_opt = st.selectbox("Nguồn dự báo:", 
                                    ["Dự báo cá nhân (Besttrack)", "Dự báo của CMA", "Dự báo của JMA"])
            if "CMA" in storm_opt: active_link = LINKS["CMA"]
            elif "JMA" in storm_opt: active_link = LINKS["JMA"]

        elif topic == "Quan trắc thời gian thực":
            # [cite: 8, 9, 10]
            obs_real = st.radio("Loại dữ liệu:", ["Số liệu radar", "Số liệu vệ tinh"])
            if "radar" in obs_real: active_link = LINKS["RADAR"]

        elif topic == "Dữ liệu quan trắc":
            # [cite: 11, 12, 13, 15]
            obs_src = st.radio("Nguồn dữ liệu:", 
                              ["Bản đồ gió (Vận hành)", "Thời tiết (WeatherObs)", "Quan trắc thời gian thực (Kttv)"])
            if "gió" in obs_src.lower(): active_link = LINKS["GIO_VAN_HANH"]
            elif "weatherobs" in obs_src.lower(): active_link = LINKS["WEATHER_OBS"]
            else: active_link = LINKS["QUANTRAC_REALTIME"]

        elif topic == "Dự báo thời tiết & khí hậu":
            # [cite: 16, 17, 19, 21]
            fore_src = st.radio("Mô hình:", ["Dự báo mô hình (GFS)", "Dự báo điểm (KMA)", "Dự báo điểm (NCHMF)"])
            if "GFS" in fore_src: active_link = LINKS["GFS_MODEL"]
            elif "KMA" in fore_src: active_link = LINKS["KMA_POINT"]
            else: active_link = LINKS["NCHMF_POINT"]

    # Hiển thị
    if active_link:
        # Tự động đăng nhập ngầm qua URL cho các trang có Auth 
        if "@" in active_link:
            st.link_button("🌐 Click để mở trang (Đã nhúng mật khẩu)", active_link)
        components.iframe(active_link, scrolling=True)
    else:
        st.info("Hệ thống đang sẵn sàng. Vui lòng chọn nguồn dữ liệu từ Sidebar.")

if __name__ == "__main__":
    main()
