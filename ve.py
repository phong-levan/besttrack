# --- TẢI RIÊNG TỪNG TỈNH ---
                    cache = st.session_state.get('interp_cache')
                    if cache and cache.get('mask_shape') is not None and not cache['mask_shape'].empty:
                        st.markdown("---")
                        st.markdown("### 🎯 Tải bản vẽ Tỉnh riêng lẻ (Cắt theo ranh giới tỉnh)")
                        
                        shape_col = cache.get('shape_col')
                        
                        # ---> THÊM ĐOẠN KIỂM TRA LỖI NÀY:
                        if shape_col is None or shape_col not in cache['mask_shape'].columns:
                            st.warning("⚠️ Không thể tải riêng từng tỉnh vì Shapefile (vn34tinh.shp) không có cột chứa tên Tỉnh. Tính năng này tạm thời bị vô hiệu hóa.")
                        else:
                            # Nếu tìm thấy cột, tiếp tục chạy tính năng như bình thường
                            available_provs = sorted(cache['mask_shape'][shape_col].dropna().unique().tolist())
                            
                            col_sel1, col_sel2 = st.columns([2, 2])
                            with col_sel1:
                                selected_dl_prov = st.selectbox("Hộp chọn Tỉnh muốn tải:", ["-- Click trên bản đồ hoặc Chọn tại đây --"] + available_provs)
                            
                            clicked_prov = None
                            if map_data and map_data.get("last_active_drawing"):
                                props = map_data["last_active_drawing"].get("properties", {})
                                if shape_col in props:
                                    clicked_prov = props[shape_col]
                                    with col_sel2:
                                        st.write("")
                                        st.info(f"💡 Đang click chọn: **{clicked_prov}** trên bản đồ.")
                            
                            final_dl_prov = None
                            if selected_dl_prov != "-- Click trên bản đồ hoặc Chọn tại đây --":
                                final_dl_prov = selected_dl_prov
                            elif clicked_prov:
                                final_dl_prov = clicked_prov
                                
                            if final_dl_prov:
                                prov_fig = generate_single_province_fig(cache, final_dl_prov, st.session_state.get("title_custom_interp", "Bản đồ Nội Suy"))
                                
                                if prov_fig:
                                    col_p1, col_p2 = st.columns([1, 3])
                                    with col_p1:
                                        fmt_prov = st.selectbox("Định dạng ảnh:", ["png", "pdf"], key="fmt_prov")
                                        
                                    buf_prov = io.BytesIO()
                                    prov_fig.savefig(buf_prov, format=fmt_prov, dpi=300, bbox_inches='tight')
                                    buf_prov.seek(0)
                                    
                                    with col_p2:
                                        st.write(""); st.write("")
                                        st.download_button(
                                            label=f"⬇️ Tải ảnh Tỉnh {final_dl_prov} ({fmt_prov.upper()})", 
                                            data=buf_prov, 
                                            file_name=f"ban_do_{final_dl_prov}.{fmt_prov}", 
                                            mime=f"image/{fmt_prov}", 
                                            key="dl_btn_prov"
                                        )
