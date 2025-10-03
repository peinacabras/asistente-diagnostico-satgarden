"""
ASISTENTE TÉCNICO SATGARDEN MVP
Stack: OpenAI (LLM + Embeddings) + Supabase (pgvector) + Streamlit
Versión completa con Recambios + Visión GPT-4
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
import json
import base64
from io import BytesIO

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

EMBEDDING_MODEL = "text-embedding-3-small"

# ============================================
# FUNCIONES DE PROCESAMIENTO DE DOCUMENTOS
# ============================================

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
        st.error(f"Error extrayendo texto: {str(e)}")
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
    
    return [c for c in chunks if len(c) > 100]

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
        st.error(f"Error guardando: {str(e)}")
        return None

def ingest_pdf(pdf_path, doc_type="manual"):
    filename = os.path.basename(pdf_path)
    st.info(f"Procesando: {filename}")
    
    text = extract_text_from_pdf(pdf_path)
    if not text:
        st.error(f"No se pudo extraer texto de {filename}")
        return
    
    st.info(f"Texto extraído: {len(text)} caracteres")
    chunks = chunk_text(text)
    
    if not chunks:
        st.error(f"No se pudieron crear chunks de {filename}")
        return
    
    st.info(f"Creados {len(chunks)} chunks")
    
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
        st.success(f"{filename}: {success_count} chunks guardados")
    else:
        st.warning(f"{filename}: {success_count}/{len(chunks)} chunks guardados")

def ingest_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        st.info(f"Procesando {len(df)} registros")
        
        progress_bar = st.progress(0)
        success_count = 0
        
        for i, row in df.iterrows():
            content = f"""
            Modelo: {row.get('modelo', 'N/A')}
            Avería: {row.get('averia', 'N/A')}
            Diagnóstico: {row.get('diagnostico', 'N/A')}
            Piezas: {row.get('piezas', 'N/A')}
            Tiempo: {row.get('tiempo_min', 'N/A')} min
            Coste: {row.get('coste_piezas', 'N/A')}
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
        
        st.success(f"{success_count}/{len(df)} casos guardados")
    except Exception as e:
        st.error(f"Error procesando CSV: {str(e)}")

# ============================================
# FUNCIONES DE BÚSQUEDA Y RESPUESTA
# ============================================

def search_similar_documents(query, top_k=5):
    try:
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            return []
        
        result = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': top_k
            }
        ).execute()
        
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Error en búsqueda: {str(e)}")
        return []

def generate_technical_response(query, context_docs, tipo_consulta):
    if context_docs:
        context = "\n\n---\n\n".join([
            f"Fuente: {doc['metadata'].get('source', 'Desconocida')}\n{doc['content']}"
            for doc in context_docs
        ])
    else:
        context = "No se encontraron documentos específicos."
    
    prompts_por_tipo = {
        "Mantenimiento": "Proporciona información de mantenimiento estructurada: Procedimiento, Tareas, Periodicidad, Herramientas, Precauciones.",
        "Recambios": "Proporciona información sobre recambios: códigos, compatibilidad, procedimiento de sustitución.",
        "Despiece": "Proporciona información sobre componentes: ubicación, códigos, secuencia de desmontaje.",
        "Procedimiento": "Proporciona procedimientos paso a paso con preparación, pasos detallados, verificación.",
        "Avería": "Diagnostica con causas probables ordenadas por probabilidad, piezas a verificar, procedimiento."
    }
    
    system_prompt = f"Eres un técnico experto de Satgarden. {prompts_por_tipo.get(tipo_consulta, prompts_por_tipo['Procedimiento'])} Sé específico y práctico."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXTO:\n{context}\n\nCONSULTA:\n{query}"}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generando respuesta: {str(e)}"

# ============================================
# FUNCIONES DE RECAMBIOS Y VISIÓN
# ============================================

def extract_parts_from_image(image_file, modelo):
    try:
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_type = image_file.type.split('/')[-1]
        
        prompt = f"""Analiza esta imagen de despiece técnico de {modelo}.

Extrae TODA la información de piezas visible:
- Número de posición
- Código de referencia
- Nombre de la pieza
- Cantidad
- Observaciones

Responde SOLO con JSON válido:

{{
  "modelo_maquina": "{modelo}",
  "piezas": [
    {{
      "numero_posicion": "1",
      "codigo": "SIS-350-001",
      "nombre": "Tornillo M8",
      "cantidad": "4",
      "observaciones": "Acero inoxidable"
    }}
  ],
  "notas_generales": "Info adicional"
}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.2
        )
        
        response_text = response.choices[0].message.content.strip()
        
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        return json.loads(response_text)
        
    except json.JSONDecodeError:
        st.error("La IA no pudo extraer las piezas. Intenta con imagen más clara.")
        return None
    except Exception as e:
        st.error(f"Error procesando imagen: {str(e)}")
        return None

def save_parts_to_catalog(piezas_data, imagen_nombre):
    try:
        success_count = 0
        modelo = piezas_data.get('modelo_maquina', 'No especificado')
        
        for pieza in piezas_data.get('piezas', []):
            data = {
                "modelo_maquina": modelo,
                "numero_pieza": str(pieza.get('numero_posicion', '')),
                "codigo_referencia": pieza.get('codigo', 'No especificado'),
                "nombre_pieza": pieza.get('nombre', ''),
                "cantidad": str(pieza.get('cantidad', '1')),
                "observaciones": pieza.get('observaciones', ''),
                "imagen_source": imagen_nombre,
                "notas_generales": piezas_data.get('notas_generales', '')
            }
            
            result = supabase.table("piezas_catalogo").insert(data).execute()
            if result:
                success_count += 1
        
        return success_count
    except Exception as e:
        st.error(f"Error guardando: {str(e)}")
        return 0

def search_parts_in_catalog(modelo=None, pieza_buscar=None):
    try:
        query = supabase.table("piezas_catalogo").select("*")
        
        if modelo:
            query = query.ilike("modelo_maquina", f"%{modelo}%")
        
        if pieza_buscar:
            query = query.or_(f"nombre_pieza.ilike.%{pieza_buscar}%,codigo_referencia.ilike.%{pieza_buscar}%")
        
        result = query.limit(50).execute()
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Error buscando: {str(e)}")
        return []

def search_parts_online(modelo, pieza_descripcion):
    try:
        prompt = f"""Información sobre repuesto:

Máquina: {modelo}
Pieza: {pieza_descripcion}

Proporciona:
1. Posibles códigos de referencia
2. Nombres alternativos
3. Proveedores habituales
4. Rango de precio estimado
5. Piezas alternativas compatibles"""

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Eres experto en repuestos de maquinaria agrícola."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================
# CALCULADORA Y PDF
# ============================================

def generate_budget_estimate(trabajo, modelo, descripcion):
    prompt = f"""Presupuesto para:
Tipo: {trabajo}
Modelo: {modelo}
Descripción: {descripcion}

Responde SOLO con JSON:
{{
  "tiempo_horas": 2.5,
  "tiempo_justificacion": "Breve explicación",
  "piezas": [{{"nombre": "Pieza", "codigo": "COD-123", "precio_estimado": 45}}],
  "herramientas_especiales": [],
  "dificultad": "Media",
  "notas_adicionales": ""
}}"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Responde SOLO JSON, sin markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        return json.loads(response_text)
    except:
        st.error("Error generando presupuesto")
        return None

def generate_pdf_report(consulta_data, respuesta, fuentes, tipo_reporte="consulta"):
    if not REPORTLAB_AVAILABLE:
        st.error("ReportLab no disponible")
        return None
    
    try:
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
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.5*cm))
        
        story.append(Paragraph("<b>Consulta:</b>", styles['Heading2']))
        story.append(Paragraph(consulta_data.get('consulta', ''), styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
        
        story.append(Paragraph("<b>Respuesta:</b>", styles['Heading2']))
        
        respuesta_limpia = respuesta.replace('#', '').replace('**', '')
        for line in respuesta_limpia.split('\n'):
            if line.strip():
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 0.2*cm))
        
        story.append(Spacer(1, 1*cm))
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
        story.append(Paragraph("Satgarden | www.satgarden.com | +34 935122686", footer_style))
        
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
    except:
        pass

# ============================================
# APLICACIÓN PRINCIPAL
# ============================================

def main():
    st.set_page_config(
        page_title="Asistente Técnico Satgarden",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Asistente Técnico Satgarden")
    st.markdown("Sistema RAG con IA")
    
    with st.sidebar:
        st.header("Administración")
        
        with st.expander("Cargar Documentos"):
            st.subheader("Manuales PDF")
            uploaded_pdfs = st.file_uploader(
                "Sube manuales",
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
                "Sube histórico",
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
        
        st.header("Estadísticas")
        try:
            doc_count = supabase.table("documents").select("id", count="exact").execute()
            st.metric("Chunks en base", doc_count.count if doc_count.count else 0)
        except:
            st.metric("Chunks en base", "Error")
    
    tabs = st.tabs(["Consulta Técnica", "Recambios", "Calculadora", "Dashboard", "Historial"])
    
    # TAB 0: CONSULTA
    with tabs[0]:
        st.header("Nueva Consulta Técnica")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            modelo = st.text_input("Modelo", placeholder="Ej: SIS-350", key="modelo")
        
        with col2:
            tipo_consulta = st.selectbox("Tipo", ["Mantenimiento", "Avería", "Recambios", "Despiece", "Procedimiento"])
        
        consulta = st.text_area("¿Qué necesitas saber?", height=120, key="consulta")
        tecnico = st.text_input("Técnico (opcional)", key="tecnico")
        
        if st.button("Buscar Información", type="primary", use_container_width=True):
            if not consulta:
                st.error("Describe tu consulta")
            else:
                query = f"Modelo: {modelo or 'No especificado'}\nTipo: {tipo_consulta}\nConsulta: {consulta}"
                
                with st.spinner("Buscando..."):
                    similar_docs = search_similar_documents(query, top_k=5)
                    respuesta = generate_technical_response(query, similar_docs, tipo_consulta)
                    log_diagnostic(tecnico or "Anónimo", modelo or "No especificado", consulta, respuesta, None)
                
                st.success("Información encontrada")
                st.markdown(respuesta)
                
                st.divider()
                
                consulta_data = {
                    'tecnico': tecnico or "Anónimo",
                    'modelo': modelo or "No especificado",
                    'tipo': tipo_consulta,
                    'consulta': consulta
                }
                
                pdf_buffer = generate_pdf_report(consulta_data, respuesta, similar_docs)
                
                if pdf_buffer:
                    st.download_button(
                        "Descargar PDF",
                        data=pdf_buffer,
                        file_name=f"consulta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    
    # TAB 1: RECAMBIOS
    with tabs[1]:
        st.header("Catálogo de Recambios")
        
        mode = st.radio("Modo:", ["Buscar pieza", "Catalogar desde despiece"], horizontal=True)
        
        if mode == "Buscar pieza":
            col1, col2 = st.columns(2)
            
            with col1:
                modelo_buscar = st.text_input("Modelo", placeholder="SIS-350", key="modelo_buscar")
            
            with col2:
                pieza_buscar = st.text_input("Pieza", placeholder="rodamiento, filtro...", key="pieza_buscar")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("Buscar en Catálogo", use_container_width=True, type="primary"):
                    if not modelo_buscar and not pieza_buscar:
                        st.warning("Introduce al menos modelo o pieza")
                    else:
                        with st.spinner("Buscando..."):
                            resultados = search_parts_in_catalog(modelo_buscar, pieza_buscar)
                        
                        if resultados:
                            st.success(f"{len(resultados)} piezas encontradas")
                            
                            df_resultados = pd.DataFrame(resultados)
                            df_display = df_resultados[['modelo_maquina', 'numero_pieza', 'codigo_referencia', 'nombre_pieza', 'cantidad']].rename(columns={
                                'modelo_maquina': 'Modelo',
                                'numero_pieza': 'N°',
                                'codigo_referencia': 'Código',
                                'nombre_pieza': 'Pieza',
                                'cantidad': 'Cant.'
                            })
                            
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                        else:
                            st.info("No se encontraron piezas en catálogo")
            
            with col_btn2:
                if st.button("Buscar en Internet", use_container_width=True):
                    if not modelo_buscar or not pieza_buscar:
                        st.warning("Necesitas modelo y pieza")
                    else:
                        with st.spinner("Buscando online..."):
                            resultado = search_parts_online(modelo_buscar, pieza_buscar)
                        
                        st.subheader("Resultados online")
                        st.markdown(resultado)
        
        else:  # Catalogar
            st.subheader("Catalogar desde Imagen")
            
            modelo_despiece = st.text_input("Modelo del despiece", placeholder="SIS-350", key="modelo_despiece")
            
            uploaded_image = st.file_uploader("Sube imagen del despiece", type=['png', 'jpg', 'jpeg'], key="despiece_uploader")
            
            if uploaded_image:
                col_img, col_preview = st.columns([1, 1])
                
                with col_img:
                    st.image(uploaded_image, use_column_width=True)
                
                with col_preview:
                    st.info("La IA extraerá códigos, nombres, posiciones y cantidades")
                
                if st.button("Extraer Piezas con IA", type="primary", use_container_width=True):
                    if not modelo_despiece:
                        st.warning("Especifica el modelo")
                    else:
                        with st.spinner("Analizando con GPT-4 Vision (10-20s)..."):
                            uploaded_image.seek(0)
                            piezas_extraidas = extract_parts_from_image(uploaded_image, modelo_despiece)
                        
                        if piezas_extraidas:
                            st.success(f"{len(piezas_extraidas.get('piezas', []))} piezas identificadas")
                            
                            if piezas_extraidas.get('notas_generales'):
                                st.info(f"Notas: {piezas_extraidas['notas_generales']}")
                            
                            piezas_list = piezas_extraidas.get('piezas', [])
                            if piezas_list:
                                df_piezas = pd.DataFrame(piezas_list)
                                
                                column_mapping = {
                                    'numero_posicion': 'N°',
                                    'codigo': 'Código',
                                    'nombre': 'Pieza',
                                    'cantidad': 'Cant.',
                                    'observaciones': 'Obs.'
                                }
                                df_display = df_piezas.rename(columns=column_mapping)
                                
                                st.dataframe(df_display, use_container_width=True, hide_index=True)
                                
                                col_save, col_download = st.columns(2)
                                
                                with col_save:
                                    if st.button("Guardar en Catálogo", use_container_width=True):
                                        saved_count = save_parts_to_catalog(piezas_extraidas, uploaded_image.name)
                                        if saved_count > 0:
                                            st.success(f"{saved_count} piezas guardadas")
                                
                                with col_download:
                                    csv = df_piezas.to_csv(index=False)
                                    st.download_button(
                                        "Descargar CSV",
                                        data=csv,
                                        file_name=f"despiece_{modelo_despiece}_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
    
    # TAB 2: CALCULADORA
    with tabs[2]:
        st.header("Calculadora de Presupuesto")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            modelo_presupuesto = st.text_input("Modelo", placeholder="SIS-350", key="modelo_presupuesto")


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
