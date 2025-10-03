"""
ASISTENTE DE DIAGNÓSTICO TÉCNICO - SATGARDEN MVP
Stack: OpenAI (LLM + Embeddings) + Supabase (pgvector) + Streamlit
Versión mejorada con mejor chunking
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

# Cargar variables de entorno
load_dotenv()

# ============================================
# CONFIGURACIÓN
# ============================================

# Clientes
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Configuración de embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# ============================================
# FUNCIONES DE INGESTIÓN DE DATOS MEJORADAS
# ============================================

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un PDF con mejor limpieza"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Limpiar caracteres extraños
                    page_text = page_text.replace('\x00', '')
                    # Normalizar espacios
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text += page_text + "\n\n"
    except Exception as e:
        st.error(f"Error extrayendo texto del PDF: {str(e)}")
        return ""
    
    return text.strip()

def chunk_text(text, chunk_size=1800, overlap=300):
    """Divide texto en chunks inteligentes respetando párrafos"""
    if not text or len(text) < 100:
        return []
    
    chunks = []
    
    # Dividir por doble salto de línea (párrafos)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Si añadir este párrafo excede el tamaño
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            # Guardar el chunk actual si no está vacío
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Si el párrafo es muy largo, dividirlo por frases
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
    
    # Añadir el último chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filtrar chunks muy cortos
    chunks = [c for c in chunks if len(c) > 100]
    
    return chunks

def generate_embedding(text):
    """Genera embedding con OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000]  # Limitar longitud por seguridad
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generando embedding: {str(e)}")
        return None

def store_document(content, metadata):
    """Guarda documento en Supabase con su embedding"""
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
    """Procesa PDF completo y lo guarda en Supabase"""
    filename = os.path.basename(pdf_path)
    st.info(f"📄 Procesando: {filename}")
    
    # Extraer texto
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        st.error(f"❌ No se pudo extraer texto de {filename}")
        return
    
    st.info(f"✓ Texto extraído: {len(text)} caracteres")
    
    # Dividir en chunks
    chunks = chunk_text(text)
    
    if not chunks:
        st.error(f"❌ No se pudieron crear chunks de {filename}")
        return
    
    st.info(f"✓ Creados {len(chunks)} chunks")
    
    # Guardar cada chunk
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
    """Procesa CSV de histórico de reparaciones"""
    try:
        df = pd.read_csv(csv_path)
        st.info(f"📊 Procesando {len(df)} registros de reparaciones")
        
        progress_bar = st.progress(0)
        success_count = 0
        
        for i, row in df.iterrows():
            # Crear texto descriptivo del registro
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

# ============================================
# FUNCIONES DE BÚSQUEDA Y DIAGNÓSTICO
# ============================================

def search_similar_documents(query, top_k=5):
    """Busca documentos similares en Supabase"""
    try:
        # Generar embedding de la query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            return []
        
        # Llamar a la función de Supabase
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
    """Genera respuesta técnica según el tipo de consulta"""
    
    # Construir contexto desde documentos recuperados
    if context_docs:
        context = "\n\n---\n\n".join([
            f"Fuente: {doc['metadata'].get('source', 'Desconocida')}\n{doc['content']}"
            for doc in context_docs
        ])
    else:
        context = "No se encontraron documentos específicos en la base de conocimiento."
    
    # Prompts especializados según tipo de consulta
    prompts_por_tipo = {
        "Mantenimiento": """Eres un técnico experto de Satgarden. Proporciona información detallada sobre mantenimiento.

FORMATO DE RESPUESTA:

## 📋 Procedimiento de Mantenimiento

[Descripción general del mantenimiento]

## 🔧 Tareas a Realizar

1. [Tarea 1 con detalles específicos]
2. [Tarea 2 con detalles específicos]
...

## ⏱️ Periodicidad Recomendada

- Frecuencia: [diaria/semanal/mensual/anual]
- Duración estimada: [tiempo]

## 🛠️ Herramientas Necesarias

- [Lista de herramientas]

## ⚠️ Precauciones

- [Aspectos de seguridad importantes]
""",
        
        "Recambios": """Eres un técnico experto de Satgarden. Proporciona información sobre recambios y piezas.

FORMATO DE RESPUESTA:

## 🔧 Recambios Necesarios

| Pieza | Código | Cantidad | Notas |
|-------|--------|----------|-------|
| [nombre] | [código] | [cant] | [info] |

## 📦 Información Adicional

- Compatibilidad: [modelos compatibles]
- Disponibilidad: [info sobre stock]
- Coste estimado: [rango de precio si disponible]

## 🔄 Procedimiento de Sustitución

[Pasos básicos para cambiar la pieza]
""",
        
        "Despiece": """Eres un técnico experto de Satgarden. Proporciona información sobre despiece y componentes.

FORMATO DE RESPUESTA:

## 🔩 Componentes Principales

1. **[Nombre componente]**
   - Código: [código]
   - Función: [descripción]
   - Ubicación: [dónde está]

2. **[Siguiente componente]**
   ...

## 📐 Diagrama/Secuencia

[Descripción del orden de desmontaje]

## ⚙️ Ensamblaje

[Secuencia de montaje inversa o específica]
""",
        
        "Procedimiento": """Eres un técnico experto de Satgarden. Proporciona procedimientos técnicos paso a paso.

FORMATO DE RESPUESTA:

## 📝 Procedimiento: [Nombre]

### Preparación

- [Requisitos previos]
- [Herramientas necesarias]

### Pasos

1. **[Paso 1]**
   - [Detalle]
   - [Precaución si aplica]

2. **[Paso 2]**
   ...

### Verificación

- [Cómo verificar que se hizo correctamente]

### Tiempo Estimado

- [Duración aproximada]
""",
        
        "Avería": """Eres un técnico experto de Satgarden. Diagnostica y proporciona soluciones.

FORMATO DE RESPUESTA:

## 🔍 Diagnósticos Probables

1. **[Causa más probable]** (80%)
   - Síntomas: [descripción]
   - Solución: [pasos]

2. **[Segunda causa]** (15%)
   ...

## 🔧 Piezas a Verificar/Cambiar

- [Lista con códigos]

## 📋 Procedimiento de Reparación

[Pasos detallados]

## ⏱️ Estimación

- Tiempo: [minutos/horas]
- Dificultad: [Baja/Media/Alta]
"""
    }
    
    system_prompt = prompts_por_tipo.get(tipo_consulta, prompts_por_tipo["Procedimiento"])
    
    system_prompt += "\n\nSé específico, práctico y cita las fuentes cuando estén disponibles. Si no tienes información suficiente, indícalo claramente."

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
    """Genera estimación de presupuesto usando IA"""
    
    prompt = f"""Eres un experto en presupuestación de trabajos de maquinaria agrícola de Satgarden.

Analiza este trabajo y proporciona una estimación realista:

Tipo de trabajo: {trabajo}
Modelo de máquina: {modelo}
Descripción: {descripcion}

Proporciona una respuesta en este formato JSON:

{{
  "tiempo_horas": 2.5,
  "tiempo_justificacion": "Breve explicación del por qué ese tiempo",
  "piezas": [
    {{"nombre": "Nombre pieza", "codigo": "COD-123", "precio_estimado": 45, "notas": "Si aplica"}},
    {{"nombre": "Otra pieza", "codigo": "COD-456", "precio_estimado": 78}}
  ],
  "herramientas_especiales": ["Si necesita algo específico"],
  "dificultad": "Baja/Media/Alta",
  "notas_adicionales": "Cualquier consideración importante"
}}

Si no conoces precios exactos de piezas, estima rangos realistas para maquinaria agrícola. Si no se necesitan piezas, deja el array vacío.
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Eres un experto en maquinaria agrícola. Responde SOLO con JSON válido, sin texto adicional."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        import json
        resultado = json.loads(response.choices[0].message.content)
        return resultado
    except Exception as e:
        st.error(f"Error generando estimación: {str(e)}")
        return None

def generate_pdf_report(consulta_data, respuesta, fuentes, tipo_reporte="consulta"):
    """Genera PDF con la consulta técnica o presupuesto"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO
    except ImportError:
        st.error("Error: La librería reportlab no está instalada correctamente")
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
    
    # Encabezado
    if tipo_reporte == "consulta":
        story.append(Paragraph("CONSULTA TÉCNICA", title_style))
    else:
        story.append(Paragraph("PRESUPUESTO", title_style))
    
    story.append(Paragraph(f"Satgarden - {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Datos de la consulta
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
    
    # Consulta/Descripción
    story.append(Paragraph("<b>Consulta:</b>", styles['Heading2']))
    story.append(Paragraph(consulta_data.get('consulta', ''), styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Respuesta
    story.append(Paragraph("<b>Respuesta Técnica:</b>", styles['Heading2']))
    
    # Convertir markdown a texto plano para PDF
    respuesta_limpia = respuesta.replace('#', '').replace('**', '').replace('*', '')
    for line in respuesta_limpia.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
    
    story.append(Spacer(1, 0.5*cm))
    
    # Fuentes consultadas
    if fuentes:
        story.append(Paragraph("<b>Fuentes Consultadas:</b>", styles['Heading2']))
        for i, doc in enumerate(fuentes[:3]):
            fuente_nombre = doc['metadata'].get('source', 'Desconocida')
            story.append(Paragraph(f"{i+1}. {fuente_nombre}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
    
    # Footer
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
    """Registra diagnóstico para análisis posterior"""
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
        st.error(f"Error registrando diagnóstico: {str(e)}")

# ============================================
# INTERFAZ STREAMLIT
# ============================================

def main():
    st.set_page_config(
        page_title="Asistente Diagnóstico Satgarden",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Asistente Técnico Satgarden")
    st.markdown("Sistema RAG con IA")
    
    # Sidebar para gestión de datos
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
                    # Guardar temporalmente
                    temp_path = f"temp_{pdf_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    
                    ingest_pdf(temp_path, doc_type="manual")
                    
                    # Limpiar archivo temporal
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
        
        # Estadísticas
        st.header("📊 Estadísticas")
        try:
            doc_count = supabase.table("documents").select("id", count="exact").execute()
            st.metric("Chunks en base", doc_count.count if doc_count.count else 0)
        except Exception as e:
            st.metric("Chunks en base", "Error")
            st.caption(str(e))
    
    # Interfaz principal
    tabs = st.tabs(["🔍 Consulta Técnica", "💰 Calculadora", "📊 Dashboard", "🔍 Búsqueda", "📝 Historial"])
    
    # TAB 0: CONSULTA TÉCNICA
    with tabs[0]:
        st.header("Nueva Consulta Técnica")
        st.caption("Busca procedimientos, mantenimiento, recambios, despieces y soluciones técnicas")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            modelo = st.text_input(
                "Modelo de máquina",
                placeholder="Ej: SIS-350, INFACO F3020, AMB Rousset CR100...",
                key="modelo",
                help="Escribe el modelo exacto de la máquina"
            )
        
        with col2:
            tipo_consulta = st.selectbox(
                "Tipo de consulta",
                ["Mantenimiento", "Avería", "Recambios", "Despiece", "Procedimiento", "Otro"]
            )
        
        consulta = st.text_area(
            "¿Qué necesitas saber?",
            placeholder="Ejemplos:\n• ¿Cuál es el procedimiento de mantenimiento anual del SIS-350?\n• ¿Qué recambios necesito para cambiar el motor?\n• ¿Cómo se desmonta la bandeja vibratoria?\n• ¿Dónde puedo encontrar el despiece del sistema hidráulico?",
            height=120,
            key="consulta"
        )
        
        tecnico = st.text_input("Técnico (opcional)", value="", key="tecnico", placeholder="Tu nombre")
        
        # Foto opcional
        uploaded_image = st.file_uploader(
            "📸 Foto de la máquina/pieza (opcional)",
            type=['png', 'jpg', 'jpeg'],
            key="image_uploader"
        )
        if uploaded_image:
            st.image(uploaded_image, width=300)
        
        if st.button("🔍 Buscar Información", type="primary", use_container_width=True):
            if not consulta:
                st.error("Por favor, describe tu consulta")
            else:
                # Construir query completa
                query = f"""
                Modelo: {modelo if modelo else 'No especificado'}
                Tipo de consulta: {tipo_consulta}
                Consulta: {consulta}
                """
                
                with st.spinner("Buscando en manuales y base de conocimiento..."):
                    # Buscar documentos relevantes
                    similar_docs = search_similar_documents(query, top_k=5)
                    
                    if not similar_docs:
                        st.warning("⚠️ No se encontraron documentos relevantes. Respuesta basada en conocimiento general.")
                    
                    # Generar respuesta
                    respuesta = generate_technical_response(query, similar_docs, tipo_consulta)
                    
                    # Registrar en log
                    try:
                        log_diagnostic(tecnico or "Anónimo", modelo or "No especificado", consulta, respuesta, None)
                    except:
                        pass  # Si falla el log, continuar de todas formas
                
                # Mostrar resultado
                st.success("✅ Información encontrada")
                st.markdown(respuesta)
                
                # Botón para descargar PDF
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
                else:
                    st.warning("No se pudo generar el PDF")
                
                # Feedback
                st.divider()
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    if st.button("👍 Información útil", key="useful"):
                        st.success("¡Gracias por el feedback!")
                with col_fb2:
                    if st.button("👎 No fue útil", key="not_useful"):
                        st.info("Feedback registrado para mejorar")
                
                # Mostrar fuentes consultadas
                if similar_docs:
                    with st.expander("📚 Fuentes consultadas"):
                        for i, doc in enumerate(similar_docs):
                            similarity = doc.get('similarity', 0) * 100
                            st.markdown(f"**Fuente {i+1}:** {doc['metadata'].get('source', 'Desconocida')} (Relevancia: {similarity:.1f}%)")
                            st.text(doc['content'][:400] + "...")
                            st.divider()
    
    # TAB 2: CALCULADORA DE PRESUPUESTO
    with tabs[3]:
        st.header("💰 Calculadora de Presupuesto")
        st.caption("Estimación automática de tiempo y costes basada en IA")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            modelo_presupuesto = st.text_input(
                "Modelo de máquina",
                placeholder="Ej: SIS-350",
                key="modelo_presupuesto"
            )
        
        with col2:
            tipo_trabajo = st.selectbox(
                "Tipo de trabajo",
                ["Mantenimiento", "Reparación", "Instalación", "Revisión", "Otro"],
                key="tipo_trabajo"
            )
        
        descripcion_trabajo = st.text_area(
            "Descripción del trabajo a realizar",
            placeholder="Ej: Cambio completo de aceite, filtros y revisión general. Incluye limpieza de componentes y verificación de tolvas.",
            height=100,
            key="desc_trabajo"
        )
        
        # Configuración de tarifas
        with st.expander("⚙️ Configuración de tarifas"):
            tarifa_hora = st.number_input(
                "Tarifa hora técnico (€)",
                min_value=20,
                max_value=100,
                value=45,
                step=5,
                key="tarifa"
            )
        
        if st.button("💰 Calcular Presupuesto", type="primary", use_container_width=True, key="calc_presup"):
            if not descripcion_trabajo:
                st.error("Describe el trabajo a realizar")
            else:
                with st.spinner("Analizando trabajo y estimando costes..."):
                    estimacion = generate_budget_estimate(tipo_trabajo, modelo_presupuesto, descripcion_trabajo)
                
                if estimacion:
                    st.success("✅ Estimación completada")
                    
                    # Cálculos
                    coste_mano_obra = estimacion['tiempo_horas'] * tarifa_hora
                    coste_piezas = sum([p['precio_estimado'] for p in estimacion.get('piezas', [])])
                    total = coste_mano_obra + coste_piezas
                    
                    # Mostrar resumen
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Tiempo estimado", f"{estimacion['tiempo_horas']} h")
                    with col_b:
                        st.metric("Mano de obra", f"{coste_mano_obra:.2f}€")
                    with col_c:
                        st.metric("TOTAL", f"{total:.2f}€", delta=f"+{coste_piezas:.2f}€ piezas")
                    
                    st.divider()
                    
                    # Desglose detallado
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔧 Mano de Obra")
                        st.write(f"**Tiempo:** {estimacion['tiempo_horas']} horas")
                        st.write(f"**Tarifa:** {tarifa_hora}€/h")
                        st.write(f"**Subtotal:** {coste_mano_obra:.2f}€")
                        st.caption(estimacion.get('tiempo_justificacion', ''))
                        
                        st.markdown("---")
                        st.write(f"**Dificultad:** {estimacion.get('dificultad', 'Media')}")
                        
                        if estimacion.get('herramientas_especiales'):
                            st.write("**Herramientas especiales:**")
                            for herr in estimacion['herramientas_especiales']:
                                st.write(f"• {herr}")
                    
                    with col2:
                        st.subheader("🔩 Piezas y Recambios")
                        
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
                            
                            st.write(f"**Subtotal piezas:** {coste_piezas:.2f}€")
                        else:
                            st.info("No se requieren piezas para este trabajo")
                    
                    # Notas adicionales
                    if estimacion.get('notas_adicionales'):
                        st.info(f"**Nota:** {estimacion['notas_adicionales']}")
                    
                    # Generar PDF del presupuesto
                    st.divider()
                    
                    presupuesto_data = {
                        'tecnico': st.session_state.get('tecnico', 'Satgarden'),
                        'modelo': modelo_presupuesto or "No especificado",
                        'tipo': tipo_trabajo,
                        'consulta': descripcion_trabajo
                    }
                    
                    # Crear texto de respuesta para PDF
                    respuesta_presupuesto = f"""
ESTIMACIÓN DE PRESUPUESTO

Tiempo estimado: {estimacion['tiempo_horas']} horas
Mano de obra: {coste_mano_obra:.2f}€ ({tarifa_hora}€/h)
Piezas y recambios: {coste_piezas:.2f}€
TOTAL: {total:.2f}€

Dificultad: {estimacion.get('dificultad', 'Media')}

Justificación del tiempo:
{estimacion.get('tiempo_justificacion', '')}
"""
                    
                    if estimacion.get('piezas'):
                        respuesta_presupuesto += "\n\nPiezas necesarias:\n"
                        for pieza in estimacion['piezas']:
                            respuesta_presupuesto += f"• {pieza['nombre']} ({pieza.get('codigo', 'N/A')}): {pieza['precio_estimado']}€\n"
                    
                    if estimacion.get('notas_adicionales'):
                        respuesta_presupuesto += f"\n\nNotas: {estimacion['notas_adicionales']}"
                    
                    pdf_buffer = generate_pdf_report(presupuesto_data, respuesta_presupuesto, [], tipo_reporte="presupuesto")
                    
                    if pdf_buffer:
                        st.download_button(
                            label="📄 Descargar Presupuesto PDF",
                            data=pdf_buffer,
                            file_name=f"presupuesto_{modelo_presupuesto}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.warning("No se pudo generar el PDF")
    
    # TAB 3: DASHBOARD
    with tabs[2]:
        st.header("📊 Dashboard de Estadísticas")
        
        try:
            # Obtener datos
            logs = supabase.table("diagnostics_log").select("*").execute()
            docs = supabase.table("documents").select("*").execute()
            
            if logs.data and len(logs.data) > 0:
                df_logs = pd.DataFrame(logs.data)
                df_logs['created_at'] = pd.to_datetime(df_logs['created_at'])
                
                # Métricas generales
                st.subheader("📈 Actividad General")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Consultas", len(df_logs))
                
                with col2:
                    ultimos_7_dias = df_logs[df_logs['created_at'] > pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)]
                    st.metric("Últimos 7 días", len(ultimos_7_dias))
                
                with col3:
                    if 'fue_util' in df_logs.columns:
                        positivas = len(df_logs[df_logs['fue_util'] == True])
                        total_con_feedback = len(df_logs[df_logs['fue_util'].notna()])
                        if total_con_feedback > 0:
                            satisfaccion = (positivas / total_con_feedback) * 100
                            st.metric("Satisfacción", f"{satisfaccion:.0f}%")
                        else:
                            st.metric("Satisfacción", "N/A")
                    else:
                        st.metric("Satisfacción", "N/A")
                
                with col4:
                    st.metric("Documentos", len(docs.data) if docs.data else 0)
                
                st.divider()
                
                # Gráficos
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("🔧 Máquinas Más Consultadas")
                    if 'modelo_maquina' in df_logs.columns:
                        top_maquinas = df_logs['modelo_maquina'].value_counts().head(5)
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
                
                # Consultas recientes
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
                
                # Fuentes más referenciadas
                st.divider()
                st.subheader("📚 Documentos Más Útiles")
                if docs.data:
                    df_docs = pd.DataFrame(docs.data)
                    if 'metadata' in df_docs.columns:
                        fuentes = df_docs['metadata'].apply(lambda x: x.get('source', 'Desconocido') if isinstance(x, dict) else 'Desconocido')
                        top_fuentes = fuentes.value_counts().head(5)
                        
                        for fuente, count in top_fuentes.items():
                            st.write(f"📄 **{fuente}**: {count} fragmentos")
                
            else:
                st.info("📊 Aún no hay suficientes consultas para generar estadísticas. Realiza algunas consultas técnicas primero.")
                
                # Mostrar al menos info de documentos
                if docs.data:
                    st.metric("Documentos cargados", len(docs.data))
        
        except Exception as e:
            st.error(f"Error cargando dashboard: {str(e)}")
    
    # TAB 4: BÚSQUEDA
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
                        with st.expander(f"Resultado {i+1} - {doc['metadata'].get('source', 'Desconocida')} (Relevancia: {similarity:.1f}%)"):
                            st.markdown(doc['content'])
                else:
                    st.warning("No se encontraron resultados")
    
    # TAB 5: HISTORIAL
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
