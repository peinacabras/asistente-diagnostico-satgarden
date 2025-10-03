"""
ASISTENTE TÉCNICO SATGARDEN MVP
Stack: OpenAI (LLM + Embeddings) + Supabase (pgvector) + Streamlit
Versión completa corregida
"""

import os
import streamlit as st
from openai import OpenAI
from supabase import create_client, Client
import PyPDF2
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import re

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    page_text = page_text.replace('\x00', '')
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text += page_text + "\n\n"
    except Exception as e:
        st.error(f"Error extrayendo texto del PDF: {str(e)}")
        return ""
    return text.strip()

def chunk_text(text, chunk_size=1800, overlap=300):
    if not text or len(text) < 100:
        return []
    
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            if len(paragraph) > chunk_size:
                sentences = re.split(r'([.!?]\s+)', paragraph)
                temp_chunk = ""
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]
                    
                    if len(temp_chunk) + len(sentence) > chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        temp_chunk += sentence
                
                current_chunk = temp_chunk
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    chunks = [c for c in chunks if len(c) > 100]
    return chunks

def generate_embedding(text):
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000]
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generando embedding: {str(e)}")
        return None

def store_document(content, metadata):
    try:
        embedding = generate_embedding(content)
        if not embedding:
            return None
        
        data = {
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }
        
        result = supabase.table("documents").insert(data).execute()
        return result
    except Exception as e:
        st.error(f"Error guardando documento: {str(e)}")
        return None

def ingest_pdf(pdf_path, doc_type="manual"):
    filename = os.path.basename(pdf_path)
    st.info(f"📄 Procesando: {filename}")
    
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        st.error(f"❌ No se pudo extraer texto de {filename}")
        return
    
    st.info(f"✓ Texto extraído: {len(text)} caracteres")
    chunks = chunk_text(text)
    
    if not chunks:
        st.error(f"❌ No se pudieron crear chunks de {filename}")
        return
    
    st.info(f"✓ Creados {len(chunks)} chunks")
    
    progress_bar = st.progress(0)
    success_count = 0
    
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": filename,
            "type": doc_type,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        
        result = store_document(chunk, metadata)
        if result:
            success_count += 1
        
        progress_bar.progress((i + 1) / len(chunks))
    
    if success_count == len(chunks):
        st.success(f"✅ {filename}: {success_count} chunks guardados correctamente")
    else:
        st.warning(f"⚠️ {filename}: {success_count}/{len(chunks)} chunks guardados")

def ingest_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        st.info(f"📊 Procesando {len(df)} registros de reparaciones")
        
        progress_bar = st.progress(0)
        success_count = 0
        
        for i, row in df.iterrows():
            content = f"""
            Modelo: {row.get('modelo', 'N/A')}
            Avería: {row.get('averia', 'N/A')}
            Diagnóstico: {row.get('diagnostico', 'N/A')}
            Piezas usadas: {row.get('piezas', 'N/A')}
            Tiempo de reparación: {row.get('tiempo_min', 'N/A')} minutos
            Coste piezas: {row.get('coste_piezas', 'N/A')}
            """
            
            metadata = {
                "source": "historico_reparaciones",
                "type": "repair_case",
                "fecha": str(row.get('fecha', '')),
                "modelo": str(row.get('modelo', ''))
            }
            
            result = store_document(content.strip(), metadata)
            if result:
                success_count += 1
            
            progress_bar.progress((i + 1) / len(df))
        
        st.success(f"✅ {success_count}/{len(df)} casos de reparación guardados")
    except Exception as e:
        st.error(f"❌ Error procesando CSV: {str(e)}")

def search_similar_documents(query, top_k=5):
    try:
        st.info(f"🔍 Generando embedding para: '{query[:50]}...'")
        
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            st.error("❌ No se pudo generar el embedding")
            return []
        
        st.info(f"✓ Embedding generado: {len(query_embedding)} dimensiones")
        
        st.info("🔍 Buscando en Supabase...")
        result = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': top_k
            }
        ).execute()
        
        st.info(f"✓ Respuesta de Supabase: {len(result.data) if result.data else 0} resultados")
        
        return result.data if result.data else []
    except Exception as e:
        st.error(f"❌ Error en búsqueda: {str(e)}")
        return []

def generate_technical_response(query, context_docs, tipo_consulta):
    if context_docs:
        context = "\n\n---\n\n".join([
            f"Fuente: {doc['metadata'].get('source', 'Desconocida')}\n{doc['content']}"
            for doc in context_docs
        ])
    else:
        context = "No se encontraron documentos específicos en la base de conocimiento."
    
    prompts_por_tipo = {
        "Mantenimiento": """Eres un técnico experto de Satgarden. Proporciona información detallada sobre mantenimiento en formato estructurado con secciones claras: Procedimiento, Tareas, Periodicidad, Herramientas y Precauciones.""",
        "Recambios": """Eres un técnico experto de Satgarden. Proporciona información sobre recambios con tabla de piezas, códigos, compatibilidad y procedimiento de sustitución.""",
        "Despiece": """Eres un técnico experto de Satgarden. Proporciona información sobre componentes, ubicación, códigos y secuencia de desmontaje.""",
        "Procedimiento": """Eres un técnico experto de Satgarden. Proporciona procedimientos paso a paso con preparación, pasos detallados, verificación y tiempo estimado.""",
        "Avería": """Eres un técnico experto de Satgarden. Diagnostica con causas probables ordenadas por probabilidad, piezas a verificar y procedimiento de reparación."""
    }
    
    system_prompt = prompts_por_tipo.get(tipo_consulta, prompts_por_tipo["Procedimiento"])
    system_prompt += "\n\nSé específico, práctico y cita las fuentes cuando estén disponibles."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXTO TÉCNICO:\n{context}\n\nCONSULTA:\n{query}"}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generando respuesta: {str(e)}"

def generate_budget_estimate(trabajo, modelo, descripcion):
    prompt = f"""Eres un experto en presupuestación de maquinaria agrícola de Satgarden.

Analiza este trabajo:

Tipo: {trabajo}
Modelo: {modelo}
Descripción: {descripcion}

IMPORTANTE: Responde ÚNICAMENTE con JSON válido, sin markdown ni texto adicional.

{{
  "tiempo_horas": 2.5,
  "tiempo_justificacion": "Explicación breve",
  "piezas": [
    {{"nombre": "Nombre pieza", "codigo": "COD-123", "precio_estimado": 45}}
  ],
  "herramientas_especiales": ["Si necesita"],
  "dificultad": "Baja",
  "notas_adicionales": "Consideraciones"
}}

Dificultad: Baja, Media o Alta. Si no necesita piezas, array vacío."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Responde SOLO con JSON válido, sin markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        import json
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        resultado = json.loads(response_text)
        return resultado
    except json.JSONDecodeError:
        st.error("La IA no devolvió JSON válido. Inténtalo de nuevo.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def generate_pdf_report(consulta_data, respuesta, fuentes, tipo_reporte="consulta"):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO
    except ImportError:
        st.error("Error: reportlab no instalado")
        return None
    
    buffer = BytesIO()
    pdf_doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story = []
    
    if tipo_reporte == "consulta":
        story.append(Paragraph("CONSULTA TÉCNICA", title_style))
    else:
        story.append(Paragraph("PRESUPUESTO", title_style))
    
    story.append(Paragraph(f"Satgarden - {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    data_table = [
        ["Técnico:", consulta_data.get('tecnico', 'N/A')],
        ["Modelo:", consulta_data.get('modelo', 'N/A')],
        ["Tipo:", consulta_data.get('tipo', 'N/A')],
    ]
    
    table = Table(data_table, colWidths=[4*cm, 12*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("<b>Consulta:</b>", styles['Heading2']))
    story.append(Paragraph(consulta_data.get('consulta', ''), styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("<b>Respuesta Técnica:</b>", styles['Heading2']))
    
    respuesta_limpia = respuesta.replace('#', '').replace('**', '').replace('*', '')
    for line in respuesta_limpia.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
    
    story.append(Spacer(1, 0.5*cm))
    
    if fuentes:
        story.append(Paragraph("<b>Fuentes Consultadas:</b>", styles['Heading2']))
        for i, doc in enumerate(fuentes[:3]):
            fuente_nombre = doc['metadata'].get('source', 'Desconocida')
            story.append(Paragraph(f"{i+1}. {fuente_nombre}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
    
    story.append(Spacer(1, 1*cm))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph("Satgarden | www.satgarden.com | +34 935122686", footer_style))
    
    try:
        pdf_doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generando PDF: {str(e)}")
        return None

def log_diagnostic(tecnico, modelo, descripcion, diagnostico, fue_util=None):
    try:
        data = {
            "tecnico": tecnico,
            "modelo_maquina": modelo,
            "descripcion_averia": descripcion,
            "diagnostico_ia": diagnostico,
            "fue_util": fue_util
        }
        supabase.table("diagnostics_log").insert(data).execute()
    except Exception as e:
        st.error(f"Error registrando: {str(e)}")

def main():
    st.set_page_config(
        page_title="Asistente Técnico Satgarden",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Asistente Técnico Satgarden")
    st.markdown("Sistema RAG con IA")
    
    with st.sidebar:
        st.header("⚙️ Administración")
        
        with st.expander("📤 Cargar Documentos"):
            st.subheader("Manuales PDF")
            uploaded_pdfs = st.file_uploader(
                "Sube manuales técnicos",
                type=['pdf'],
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            if uploaded_pdfs and st.button("Procesar PDFs"):
                for pdf_file in uploaded_pdfs:
                    temp_path = f"temp_{pdf_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    
                    ingest_pdf(temp_path, doc_type="manual")
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            
            st.divider()
            
            st.subheader("Histórico CSV")
            uploaded_csv = st.file_uploader(
                "Sube histórico de reparaciones",
                type=['csv'],
                key="csv_uploader"
            )
            if uploaded_csv and st.button("Procesar CSV"):
                temp_path = f"temp_{uploaded_csv.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_csv.getbuffer())
                
                ingest_csv(temp_path)
                
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        st.divider()
        
        st.header("📊 Estadísticas")
        try:
            doc_count = supabase.table("documents").select("id", count="exact").execute()
            st.metric("Chunks en base", doc_count.count if doc_count.count else 0)
        except Exception as e:
            st.metric("Chunks en base", "Error")
    
    tabs = st.tabs(["🔍 Consulta Técnica", "🔍 Búsqueda", "💰 Calculadora", "📊 Dashboard", "📝 Historial"])
    
    # TAB 0: CONSULTA TÉCNICA
    with tabs[0]:
        st.header("Nueva Consulta Técnica")
        st.caption("Busca procedimientos, mantenimiento, recambios, despieces y soluciones técnicas")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            modelo = st.text_input(
                "Modelo de máquina",
                placeholder="Ej: SIS-350, INFACO F3020...",
                key="modelo"
            )
        
        with col2:
            tipo_consulta = st.selectbox(
                "Tipo de consulta",
                ["Mantenimiento", "Avería", "Recambios", "Despiece", "Procedimiento", "Otro"]
            )
        
        consulta = st.text_area(
            "¿Qué necesitas saber?",
            placeholder="Ejemplos:\n• ¿Cuál es el procedimiento de mantenimiento anual?\n• ¿Qué recambios necesito para cambiar el motor?\n• ¿Cómo se desmonta la bandeja vibratoria?",
            height=120,
            key="consulta"
        )
        
        tecnico = st.text_input("Técnico (opcional)", value="", key="tecnico", placeholder="Tu nombre")
        
        if st.button("🔍 Buscar Información", type="primary", use_container_width=True):
            if not consulta:
                st.error("Por favor, describe tu consulta")
            else:
                query = f"Modelo: {modelo if modelo else 'No especificado'}\nTipo de consulta: {tipo_consulta}\nConsulta: {consulta}"
                
                with st.spinner("Buscando en manuales y base de conocimiento..."):
                    similar_docs = search_similar_documents(query, top_k=5)
                    
                    if not similar_docs:
                        st.warning("⚠️ No se encontraron documentos relevantes.")
                    
                    respuesta = generate_technical_response(query, similar_docs, tipo_consulta)
                    
                    try:
                        log_diagnostic(tecnico or "Anónimo", modelo or "No especificado", consulta, respuesta, None)
                    except:
                        pass
                
                st.success("✅ Información encontrada")
                st.markdown(respuesta)
                
                st.divider()
                
                consulta_data = {
                    'tecnico': tecnico or "Anónimo",
                    'modelo': modelo or "No especificado",
                    'tipo': tipo_consulta,
                    'consulta': consulta
                }
                
                pdf_buffer = generate_pdf_report(consulta_data, respuesta, similar_docs, tipo_reporte="consulta")
                
                if pdf_buffer:
                    st.download_button(
                        label="📄 Descargar Informe PDF",
                        data=pdf_buffer,
                        file_name=f"consulta_tecnica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                st.divider()
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    if st.button("👍 Información útil", key="useful"):
                        st.success("¡Gracias!")
                with col_fb2:
                    if st.button("👎 No fue útil", key="not_useful"):
                        st.info("Feedback registrado")
                
                if similar_docs:
                    with st.expander("📚 Fuentes consultadas"):
                        for i, doc in enumerate(similar_docs):
                            similarity = doc.get('similarity', 0) * 100
                            st.markdown(f"**Fuente {i+1}:** {doc['metadata'].get('source', 'Desconocida')} ({similarity:.1f}%)")
                            st.text(doc['content'][:400] + "...")
                            st.divider()
    
    # TAB 1: BÚSQUEDA
    with tabs[1]:
        st.header("Búsqueda en Base de Conocimiento")
        
        search_query = st.text_input(
            "¿Qué información buscas?",
            placeholder="Ej: procedimiento cambio aceite SIS-350",
            key="search_query"
        )
        
        if st.button("Buscar", key="search_button"):
            if search_query:
                with st.spinner("Buscando..."):
                    results = search_similar_documents(search_query, top_k=10)
                
                if results:
                    st.write(f"**{len(results)} resultados encontrados**")
                    
                    for i, doc in enumerate(results):
                        similarity = doc.get('similarity', 0) * 100
                        with st.expander(f"Resultado {i+1} - {doc['metadata'].get('source', 'Desconocida')} ({similarity:.1f}%)"):
                            st.markdown(doc['content'])
                else:
                    st.warning("No se encontraron resultados")
    
    # TAB 2: CALCULADORA
    with tabs[2]:
        st.header("💰 Calculadora de Presupuesto")
        st.caption("Estimación automática de tiempo y costes basada en IA")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            modelo_presupuesto = st.text_input("Modelo", placeholder="Ej: SIS-350", key="modelo_presupuesto")
        
        with col2:
            tipo_trabajo = st.selectbox("Tipo", ["Mantenimiento", "Reparación", "Instalación", "Revisión", "Otro"], key="tipo_trabajo")
        
        descripcion_trabajo = st.text_area(
            "Descripción del trabajo",
            placeholder="Ej: Cambio completo de aceite, filtros y revisión general.",
            height=100,
            key="desc_trabajo"
        )
        
        with st.expander("⚙️ Configuración"):
            tarifa_hora = st.number_input("Tarifa hora técnico (€)", min_value=20, max_value=100, value=45, step=5, key="tarifa")
        
        if st.button("💰 Calcular Presupuesto", type="primary", use_container_width=True, key="calc_presup"):
            if not descripcion_trabajo:
                st.error("Describe el trabajo")
            else:
                with st.spinner("Analizando..."):
                    estimacion = generate_budget_estimate(tipo_trabajo, modelo_presupuesto, descripcion_trabajo)
                
                if estimacion:
                    st.success("✅ Estimación completada")
                    
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
                        st.subheader("🔧 Mano de Obra")
                        st.write(f"**Tiempo:** {estimacion['tiempo_horas']} h")
                        st.write(f"**Tarifa:** {tarifa_hora}€/h")
                        st.write(f"**Subtotal:** {coste_mano_obra:.2f}€")
                        st.caption(estimacion.get('tiempo_justificacion', ''))
                        st.write(f"**Dificultad:** {estimacion.get('dificultad', 'Media')}")
                    
                    with col2:
                        st.subheader("🔩 Piezas")
                        
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
        st.header("📊 Dashboard de Estadísticas")
        
        try:
            logs = supabase.table("diagnostics_log").select("*").execute()
            docs = supabase.table("documents").select("*").execute()
            
            if logs.data and len(logs.data) > 0:
                df_logs = pd.DataFrame(logs.data)
                df_logs['created_at'] = pd.to_datetime(df_logs['created_at'])
                
                st.subheader("📈 Actividad General")
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
                    st.subheader("🔧 Máquinas Más Consultadas")
                    if 'modelo_maquina' in df_logs.columns:
                        top_maquinas = df_logs['modelo_maquina'].value_counts().head(5)
                        st.bar_chart(top_maquinas)_maquinas = df_logs['modelo_maquina'].value_counts().head(5)
                        st.bar_chart(top_maquinas)
                    else:
                        st.info("No hay datos suficientes")
                
                with col_right:
                    st.subheader("👤 Técnicos Más Activos")
                    if 'tecnico' in df_logs.columns:
                        top_tecnicos = df_logs['tecnico'].value_counts().head(5)
                        st.bar_chart(top_tecnicos)
                    else:
                        st.info("No hay datos suficientes")
                
                st.divider()
                
                st.subheader("🕐 Consultas Recientes")
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
                st.info("📊 Aún no hay suficientes consultas. Realiza algunas consultas primero.")
                
                if docs.data:
                    st.metric("Documentos cargados", len(docs.data))
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
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
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
