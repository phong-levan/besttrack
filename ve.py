# ... (Giữ nguyên các phần import và hàm xử lý bên trên) ...

def main():
    with st.sidebar:
        st.title("🎛️ ĐIỀU KHIỂN")
        
        # 1. CẤU HÌNH REAL-TIME
        st.sidebar.markdown("### ⏱️ Cấu hình")
        auto_refresh = st.sidebar.checkbox("🔄 Tự động cập nhật (10p)", value=False)
        if auto_refresh:
            components.html("""<script>setTimeout(function(){window.location.reload();}, 600000);</script>""", height=0, width=0)

        st.markdown("---")
        # 2. CHỌN CHỦ ĐỀ
        topic = st.selectbox("1. CHỦ ĐỀ CHÍNH:", ["Bão (Typhoon)", "Thời tiết (Weather)", "Vệ tinh (Windy)"])
        st.markdown("---")
        
        final_df = pd.DataFrame()
        dashboard_title = ""
        show_widgets = False
        active_mode = ""

        # ... (Giữ nguyên phần logic xử lý file Excel và các Option 1, 2, 3...) ...
        # (Để ngắn gọn, bạn giữ nguyên logic đọc file Excel ở đoạn này trong code cũ nhé)
        # -----------------------------------------------------------------------
        # --- HÀM ĐỌC FILE (Copy lại từ code cũ) ---
        def process_excel(f_path):
            if not f_path or not os.path.exists(f_path): return pd.DataFrame()
            try:
                df = pd.read_excel(f_path)
                df = normalize_columns(df)
                for c in ['wind_kt', 'bf', 'r6', 'r10', 'rc']: 
                    if c not in df.columns: df[c] = 0
                if 'datetime_str' in df.columns: df['dt'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
                elif all(c in df.columns for c in ['year','mon','day','hour']): df['dt'] = pd.to_datetime(dict(year=df.year, month=df.mon, day=df.day, hour=df.hour), errors='coerce')
                for c in ['lat','lon','wind_kt']: df[c] = pd.to_numeric(df[c], errors='coerce')
                return df.dropna(subset=['lat','lon'])
            except: return pd.DataFrame()

        if topic == "Bão (Typhoon)":
            storm_opt = st.radio("2. CHỨC NĂNG:", ["Option 1: Hiện trạng", "Option 2: Lịch sử"])
            active_mode = storm_opt
            st.markdown("---")
            if "Option 1" in storm_opt:
                dashboard_title = "TIN BÃO HIỆN TẠI"
                if st.checkbox("Hiển thị lớp Hiện trạng", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack.xlsx", type="xlsx", key="o1")
                    path = f if f else (FILE_OPT1 if os.path.exists(FILE_OPT1) else None)
                    df = process_excel(path)
                    if not df.empty:
                        if 'storm_no' in df.columns:
                            all_s = df['storm_no'].unique()
                            sel = st.multiselect("Chọn bão:", all_s, default=all_s)
                            final_df = df[df['storm_no'].isin(sel)]
                        else: final_df = df
                    else: st.warning("Vui lòng tải file.")
            else: # Option 2
                dashboard_title = "LỊCH SỬ BÃO"
                if st.checkbox("Hiển thị lớp Lịch sử", value=True):
                    show_widgets = True
                    f = st.file_uploader("Upload besttrack_capgio.xlsx", type="xlsx", key="o2")
                    path = f if f else (FILE_OPT2 if os.path.exists(FILE_OPT2) else None)
                    df = process_excel(path)
                    if not df.empty:
                        years = st.multiselect("Năm:", sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
                        temp = df[df['year'].isin(years)]
                        names = st.multiselect("Tên bão:", temp['name'].unique(), default=temp['name'].unique())
                        final_df = temp[temp['name'].isin(names)]
                    else: st.warning("Vui lòng tải file.")

        elif topic == "Thời tiết (Weather)":
            weather_source = st.radio("2. NGUỒN DỮ LIỆU:", ["Option 3: Quan trắc", "Option 4: Mô hình"])
            st.markdown("---")
            w_param = st.radio("3. THÔNG SỐ:", ["Nhiệt độ", "Lượng mưa", "Gió"])
            if st.checkbox("Hiển thị lớp dữ liệu", value=True):
                show_widgets = True
                dashboard_title = f"BẢN ĐỒ {str(w_param).upper()}"

        elif topic == "Vệ tinh (Windy)":
            st.info("📡 Đang kết nối vệ tinh Windy (Real-time)...")
            windy_url = "https://embed.windy.com/embed2.html?lat=16.0&lon=114.0&detailLat=16.0&detailLon=114.0&width=1000&height=800&zoom=5&level=surface&overlay=satellite&product=satellite&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
            components.iframe(windy_url, height=1000, scrolling=False)
            return

    # --- KHỞI TẠO BẢN ĐỒ (ĐÃ SỬA: ĐƯA VỆ TINH LÊN ƯU TIÊN) ---
    m = folium.Map(location=[16.0, 114.0], zoom_start=6, tiles=None, zoom_control=False)
    
    # 1. LỚP VỆ TINH NỀN (ESRI) - Sẽ hiển thị đầu tiên trong list control
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri Satellite', 
        name='🛰️ Vệ tinh (Nền)', 
        overlay=False,
        control=True
    ).add_to(m)

    # 2. CÁC LỚP BẢN ĐỒ KHÁC
    folium.TileLayer('CartoDB positron', name='Bản đồ Sáng', overlay=False).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Chi tiết', overlay=False).add_to(m)
    
    # 3. LỚP MÂY VỆ TINH REAL-TIME (RAINVIEWER) - ĐẶT SHOW=TRUE
    latest_ts = get_rainviewer_ts()
    if latest_ts:
        st.sidebar.success(f"✅ Mây vệ tinh: Cập nhật lúc {latest_ts}")
        folium.TileLayer(
            tiles=f"https://tile.rainviewer.com/{latest_ts}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
            attr="RainViewer", 
            name="☁️ Mây Vệ tinh (Real-time)", 
            overlay=True, 
            show=True,  # <--- QUAN TRỌNG: Mặc định BẬT
            opacity=0.7
        ).add_to(m)
    else:
        st.sidebar.warning("⚠️ Không lấy được dữ liệu mây RainViewer")

    fg_storm = folium.FeatureGroup(name="🌀 Lớp Bão")
    fg_weather = folium.FeatureGroup(name="🌦️ Lớp Thời Tiết")

    # ... (Phần logic vẽ Bão & Thời tiết giữ nguyên như cũ) ...
    if not final_df.empty and topic == "Bão (Typhoon)" and show_widgets:
        if "Option 1" in str(active_mode):
            groups = final_df['storm_no'].unique() if 'storm_no' in final_df.columns else [None]
            for g in groups:
                sub = final_df[final_df['storm_no']==g] if g else final_df
                dense = densify_track(sub)
                f6, f10, fc = create_storm_swaths(dense)
                for geom, c, o in [(f6,COL_R6,0.4), (f10,COL_R10,0.5), (fc,COL_RC,0.6)]:
                    if geom and not geom.is_empty: folium.GeoJson(mapping(geom), style_function=lambda x,c=c,o=o: {'fillColor':c,'color':c,'weight':0,'fillOpacity':o}).add_to(fg_storm)
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='white', weight=2).add_to(fg_storm) # Đổi màu đường thành trắng cho nổi trên nền vệ tinh
                for _, r in sub.iterrows():
                    icon_path = os.path.join(ICON_DIR, f"{get_icon_name(r)}.png")
                    if os.path.exists(icon_path): folium.Marker([r['lat'],r['lon']], icon=folium.CustomIcon(icon_path, icon_size=(30,30))).add_to(fg_storm)
                    else: folium.CircleMarker([r['lat'],r['lon']], radius=3, color='yellow').add_to(fg_storm)
        else: 
            for n in final_df['name'].unique():
                sub = final_df[final_df['name']==n].sort_values('dt')
                folium.PolyLine(sub[['lat','lon']].values.tolist(), color='cyan', weight=2).add_to(fg_storm) # Màu cyan cho nổi
                for _, r in sub.iterrows():
                    c = '#00CCFF' if r.get('wind_kt',0)<34 else ('#FFFF00' if r.get('wind_kt',0)<64 else '#FF0000')
                    folium.CircleMarker([r['lat'],r['lon']], radius=4, color=c, fill=True, fill_opacity=1, popup=f"{n}").add_to(fg_storm)

    if topic == "Thời tiết (Weather)" and show_widgets:
        folium.Circle([16, 112], radius=100000, color='orange', fill=True, fill_opacity=0.3, popup="Vùng giả lập").add_to(fg_weather)

    fg_storm.add_to(m)
    fg_weather.add_to(m)
    
    # LAYER CONTROL (LUÔN MỞ ĐỂ BẠN THẤY)
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)

    if show_widgets:
        if not final_df.empty: st.markdown(create_info_table(final_df, dashboard_title), unsafe_allow_html=True)
        elif topic == "Thời tiết (Weather)": st.markdown(create_info_table(pd.DataFrame(), dashboard_title), unsafe_allow_html=True)
        if "Option 1" in str(active_mode) and os.path.exists(CHUTHICH_IMG):
            with open(CHUTHICH_IMG, "rb") as f: b64 = base64.b64encode(f.read()).decode()
            st.markdown(create_legend(b64), unsafe_allow_html=True)

    st_folium(m, width=None, height=1000, use_container_width=True)

if __name__ == "__main__":
    main()
