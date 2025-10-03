with col2:
            tipo_trabajo = st.selectbox("Tipo", ["Mantenimiento", "Reparación", "Instalación", "Revisión"], key="tipo_trabajo")
        
        descripcion_trabajo = st.text_area("Descripción del trabajo", height=100, key="desc_trabajo")
        
        with st.expander("Configuración"):
            tarifa_hora = st.number_input("Tarifa hora (€)", min_value=20, max_value=100, value=45, step=5, key="tarifa")
        
        if st.button("Calcular Presupuesto", type="primary", use_container_width=True):
            if not descripcion_trabajo:
                st.error("Describe el trabajo")
            else:
                with st.spinner("Analizando..."):
                    estimacion = generate_budget_estimate(tipo_trabajo, modelo_presupuesto, descripcion_trabajo)
                
                if estimacion:
                    st.success("Estimación completada")
                    
                    coste_mano_obra = estimacion['tiempo_horas'] * tarifa_hora
                    coste_piezas = sum([p['precio_estimado'] for p in estimacion.get('piezas', [])])
                    total = coste_mano_obra + coste_piezas
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Tiempo", f"{estimacion['tiempo_horas']} h")
                    with col_b:
                        st.metric("Mano de obra", f"{coste_mano_obra:.2f}€")
                    with col_c:
                        st.metric("TOTAL", f"{total:.2f}€")
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Mano de Obra")
                        st.write(f"**Tiempo:** {estimacion['tiempo_horas']} h")
                        st.write(f"**Tarifa:** {tarifa_hora}€/h")
                        st.write(f"**Subtotal:** {coste_mano_obra:.2f}€")
                        st.caption(estimacion.get('tiempo_justificacion', ''))
                        st.write(f"**Dificultad:** {estimacion.get('dificultad', 'Media')}")
                    
                    with col2:
                        st.subheader("Piezas")
                        
                        if estimacion.get('piezas'):
                            piezas_data = []
                            for pieza in estimacion['piezas']:
                                piezas_data.append({
                                    'Pieza': pieza['nombre'],
                                    'Código': pieza.get('codigo', '-'),
                                    'Precio': f"{pieza['precio_estimado']}€"
                                })
                            
                            df_piezas = pd.DataFrame(piezas_data)
                            st.dataframe(df_piezas, use_container_width=True, hide_index=True)
                            st.write(f"**Subtotal:** {coste_piezas:.2f}€")
                        else:
                            st.info("No se requieren piezas")
    
    # TAB 3: DASHBOARD
    with tabs[3]:
        st.header("Dashboard de Estadísticas")
        
        try:
            logs = supabase.table("diagnostics_log").select("*").execute()
            docs = supabase.table("documents").select("*").execute()
            
            if logs.data and len(logs.data) > 0:
                df_logs = pd.DataFrame(logs.data)
                df_logs['created_at'] = pd.to_datetime(df_logs['created_at'])
                
                st.subheader("Actividad General")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Consultas", len(df_logs))
                
                with col2:
                    ultimos_7_dias = df_logs[df_logs['created_at'] > pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)]
                    st.metric("Últimos 7 días", len(ultimos_7_dias))
                
                with col3:
                    st.metric("Satisfacción", "N/A")
                
                with col4:
                    st.metric("Documentos", len(docs.data) if docs.data else 0)
                
                st.divider()
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("Máquinas Más Consultadas")
                    if 'modelo_maquina' in df_logs.columns:
                        top_maquinas = df_logs['modelo_maquina'].value_counts().head(5)
                        st.bar_chart(top_maquinas)
                    else:
                        st.info("No hay datos")
                
                with col_right:
                    st.subheader("Técnicos Más Activos")
                    if 'tecnico' in df_logs.columns:
                        top_tecnicos = df_logs['tecnico'].value_counts().head(5)
                        st.bar_chart(top_tecnicos)
                    else:
                        st.info("No hay datos")
                
                st.divider()
                
                st.subheader("Consultas Recientes")
                df_recientes = df_logs.sort_values('created_at', ascending=False).head(10)
                st.dataframe(
                    df_recientes[['created_at', 'tecnico', 'modelo_maquina', 'descripcion_averia']].rename(columns={
                        'created_at': 'Fecha',
                        'tecnico': 'Técnico',
                        'modelo_maquina': 'Modelo',
                        'descripcion_averia': 'Consulta'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
            else:
                st.info("Aún no hay suficientes consultas para mostrar estadísticas.")
                
                if docs.data:
                    st.metric("Documentos cargados", len(docs.data))
        
        except Exception as e:
            st.error(f"Error cargando dashboard: {str(e)}")
    
    # TAB 4: HISTORIAL
    with tabs[4]:
        st.header("Historial de Diagnósticos")
        
        try:
            logs = supabase.table("diagnostics_log")\
                .select("*")\
                .order("created_at", desc=True)\
                .limit(20)\
                .execute()
            
            if logs.data:
                df_logs = pd.DataFrame(logs.data)
                st.dataframe(
                    df_logs[['created_at', 'tecnico', 'modelo_maquina', 'fue_util']],
                    use_container_width=True
                )
            else:
                st.info("Aún no hay diagnósticos registrados")
        except Exception as e:
            st.error(f"Error cargando historial: {str(e)}")

if __name__ == "__main__":
    main()
