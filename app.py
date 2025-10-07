"""
ASISTENTE TÉCNICO SATGARDEN V2.8
Implementación completa de todas las funcionalidades:
- NUEVO: Rediseño completo de la interfaz de usuario (UI/UX).
- Nueva paleta de colores, tipografía mejorada y rediseño de componentes.
- Iconos en el menú principal para una mejor usabilidad.
- Pantalla principal a modo de Hub para una navegación intuitiva.
- Informes de cierre y descarga en PDF para el módulo CMMS.
- Módulo de Gestión de Casos (Mini-CMMS) con tablero Kanban.
- Integración para crear casos directamente desde las consultas.
- Dashboard de Inteligencia Técnica mejorado y optimizado.
- Sistema de Conocimiento Verificado
- Generador de Planes de Mantenimiento Preventivo con Exportación a PDF
- Gestión de la Base de Conocimiento (Carga y Eliminación)
- Calculadora de Estimaciones
- Descarga de Consultas en PDF
"""

import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from supabase import create_client, Client
import PyPDF2
from datetime import datetime
from dotenv import load_dotenv
import re
import json
from io import BytesIO

# --- Dependencias Opcionales para PDF ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- Configuración Inicial ---
load_dotenv()
st.set_page_config(page_title="Asistente Satgarden V2.8", page_icon="🛠️", layout="wide")

# --- Estilos CSS Personalizados ---
def load_css():
    st.markdown("""
    <style>
        /* Base and Background */
        .stApp {
            background-color: #0c111e; /* Dark blue-grey background */
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #e0e0e0; /* Off-white for titles */
        }
        .stMarkdown, .stTextInput, .stTextArea, .stSelectbox {
            color: #c0c0c0; /* Lighter grey for body text */
        }

        /* --- Hub Page Specific Styles --- */
        .stButton > button {
            all: unset; /* Reset Streamlit's default button styles */
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            width: 100%;
            height: 150px; /* Fixed height for buttons */
            padding: 20px;
            font-size: 1.1em;
            font-weight: 600;
            color: #e0e0e0;
            background-color: #1c2a4a;
            border-radius: 12px;
            border: 1px solid #3a4a6a;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background-color: #2a3a5a;
            border-color: #537895;
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .stButton > button:active {
            transform: translateY(0px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .stButton > button span {
            font-size: 2.5em; /* Icon size */
            margin-bottom: 10px;
        }

        /* --- Kanban Board (CMMS) Styles --- */
        [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
            background-color: rgba(42, 58, 90, 0.3);
            border-radius: 10px;
            padding: 15px;
        }

        .kanban-header h3 {
            text-align: center;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 8px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 1.3em;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #537895; /* Accent color for headers */
        }

        /* Kanban Card Style */
        .st-emotion-cache-12w0qpk {
            background-color: #1c2a4a;
            border: 1px solid #3a4a6a;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            transition: background-color 0.2s;
        }
        .st-emotion-cache-12w0qpk:hover {
             background-color: #2a3a5a;
        }

        /* --- Sidebar Styles --- */
        .st-emotion-cache-16txtl3 {
            background-color: rgba(12, 17, 30, 0.8);
            backdrop-filter: blur(5px);
        }
    </style>
    """, unsafe_allow_html=True)

# --- Conexiones (Cacheado para Rendimiento) ---
@st.cache_resource
def init_connections():
    if not all([os.getenv("OPENAI_API_KEY"), os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")]):
        st.error("Faltan variables de entorno. Revisa tu archivo .env o los secrets en Streamlit Cloud.")
        st.stop()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return openai_client, supabase_client

openai_client, supabase = init_connections()
EMBEDDING_MODEL = "text-embedding-3-small"


# --- Funciones de Ingesta y Procesamiento de Documentos ---
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file_bytes)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += re.sub(r'\s+', ' ', page_text.replace('\x00', '')) + "\n\n"
    except Exception as e:
        st.error(f"Error extrayendo texto del PDF: {e}")
    return text.strip()

def chunk_text(text, chunk_size=2000, overlap=200):
    if not text or len(text) < 100: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 100]

def store_document_chunk(content, metadata):
    embedding = generate_embedding(content)
    if embedding:
        try:
            supabase.table("documents").insert({
                "content": content, "metadata": metadata, "embedding": embedding
            }).execute()
            return True
        except Exception as e:
            st.error(f"Error al guardar chunk: {e}")
    return False

def ingest_pdf_files(files):
    for pdf in files:
        st.info(f"Procesando: {pdf.name}...")
        file_bytes = BytesIO(pdf.getvalue())
        text = extract_text_from_pdf(file_bytes)
        if not text:
            st.warning(f"No se pudo extraer texto de {pdf.name}.")
            continue
        chunks = chunk_text(text)
        progress_bar = st.progress(0, text=f"Guardando chunks de {pdf.name}")
        for i, chunk in enumerate(chunks):
            metadata = {"source": pdf.name, "type": "manual", "chunk_index": i}
            store_document_chunk(chunk, metadata)
            progress_bar.progress((i + 1) / len(chunks))
        st.success(f"¡{pdf.name} procesado y guardado en la base de conocimiento!")
    get_dashboard_data.clear()


# --- Funciones de IA y Lógica de Negocio ---
def generate_embedding(text):
    try:
        text = text.replace("\n", " ").strip()[:8191]
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Error al generar embedding: {e}")
        return None

def generate_technical_response(query, context_docs, tipo):
    context = "\n\n---\n\n".join([f"Fuente: {doc['metadata'].get('source', 'Desconocida')}\n{doc['content']}" for doc in context_docs])
    system_prompts = {
        "Avería": "Actúa como un técnico experto de Satgarden. Tu misión es diagnosticar averías. Proporciona una lista de posibles causas por orden de probabilidad y los pasos para verificar cada una. Sé claro y directo.",
        "Mantenimiento": "Actúa como un técnico de Satgarden. Proporciona una checklist detallada de las tareas de mantenimiento solicitadas, basándote en la información de los manuales.",
        "Recambios": "Actúa como un especialista en recambios de Satgarden. Extrae los códigos de referencia y nombres de las piezas solicitadas. Si no los encuentras, sugiere cómo identificarlas.",
        "Despiece": "Actúa como un técnico de Satgarden. Describe los componentes del despiece solicitado, sus ubicaciones y relaciones entre ellos, basándote en los manuales."
    }
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompts.get(tipo, "Eres un asistente técnico de Satgarden.")},
                {"role": "user", "content": f"Basándote EXCLUSIVAMENTE en el siguiente contexto de la base de conocimiento, responde a la consulta del usuario.\n\nCONTEXTO:\n{context}\n\nCONSULTA:\n{query}"}
            ],
            temperature=0.1, max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al generar respuesta: {str(e)}"

def generate_maintenance_plan(modelo, horas):
    query = f"Manual de mantenimiento para {modelo}"
    docs = search_document_knowledge(query, top_k=10)
    if not docs:
        return "No se encontraron manuales o información relevante para ese modelo en la base de conocimiento."
    context = "\n\n---\n\n".join([doc['content'] for doc in docs])
    prompt = f"""
    CONTEXTO DE MANUALES PARA {modelo}:
    {context}

    TAREA:
    Actúa como el Director Técnico de Satgarden. Basándote en el contexto proporcionado, genera un plan de mantenimiento preventivo profesional y detallado para la máquina '{modelo}' que tiene aproximadamente {horas} horas de uso.
    El plan debe estar formateado como una checklist en Markdown.
    Si no encuentras un plan exacto para esas horas, busca el intervalo de mantenimiento más cercano (ej. 500h, 1000h) y úsalo.
    Si encuentras tareas de diferentes intervalos que deberían haberse hecho, agrúpalas por el intervalo correspondiente.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generando el plan de mantenimiento: {e}"

def generate_budget_estimate(trabajo, modelo, desc):
    context_query = f"Historial de reparaciones o mantenimientos para {modelo} sobre {desc}"
    docs = search_document_knowledge(context_query, top_k=3)
    context = "\n\n---\n\n".join([doc['content'] for doc in docs]) if docs else "No hay historial relevante."
    prompt = f"""
    CONTEXTO DE TRABAJOS SIMILARES:
    {context}

    TAREA:
    Actúa como un perito técnico experto. Analiza la siguiente solicitud de trabajo y el contexto de casos pasados.
    - Trabajo: {trabajo}
    - Máquina: {modelo}
    - Descripción: {desc}

    Devuelve tu estimación únicamente en formato JSON, sin texto adicional. El JSON debe tener la siguiente estructura:
    {{
        "tiempo_horas": float,
        "justificacion_tiempo": "string",
        "dificultad": "string (Baja, Media, Alta, Muy Alta)"
    }}
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are an expert technical estimator that only responds in JSON."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error al generar estimación: {e}")
        return None

# --- Funciones de Base de Datos (Supabase) ---
@st.cache_data(ttl=600)
def get_dashboard_data():
    try:
        response = supabase.table("diagnostics_log").select("*").execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('Europe/Madrid')
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar los datos del dashboard: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_work_orders():
    try:
        response = supabase.table("work_orders").select("*").order("created_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error al cargar los casos de trabajo: {e}")
        return []

def create_work_order(title, description, machine_model, work_type, assigned_to, priority, related_consultation_id=None):
    try:
        supabase.table("work_orders").insert({
            "title": title, "description": description, "machine_model": machine_model,
            "work_type": work_type, "assigned_to": assigned_to, "priority": priority,
            "related_consultation_id": related_consultation_id
        }).execute()
        st.success("¡Nuevo caso de trabajo creado con éxito!")
        get_work_orders.clear()
    except Exception as e:
        st.error(f"Error al crear el caso de trabajo: {e}")

def update_work_order_status(order_id, new_status):
    try:
        supabase.table("work_orders").update({"status": new_status}).eq("id", order_id).execute()
        st.toast(f"Caso #{order_id} actualizado a '{new_status}'")
        get_work_orders.clear()
    except Exception as e:
        st.error(f"Error al actualizar el estado del caso: {e}")

def update_work_order_with_report(order_id, report_text):
    try:
        supabase.table("work_orders").update({
            "status": "Cerrado",
            "final_report": report_text
        }).eq("id", order_id).execute()
        st.toast(f"Caso #{order_id} finalizado y reporte guardado.")
        get_work_orders.clear()
    except Exception as e:
        st.error(f"Error al guardar el informe del caso: {e}")

def search_verified_knowledge(query_text, top_k=1, threshold=0.8):
    embedding = generate_embedding(query_text)
    if not embedding: return None
    try:
        result = supabase.rpc('match_verified_documents', {'query_embedding': embedding, 'match_count': top_k, 'match_threshold': threshold}).execute()
        return result.data[0] if result.data else None
    except Exception:
        return None

def search_document_knowledge(query_text, top_k=5):
    embedding = generate_embedding(query_text)
    if not embedding: return []
    try:
        result = supabase.rpc('match_documents', {'query_embedding': embedding, 'match_count': top_k}).execute()
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Error en la búsqueda de documentos: {e}")
        return []

def get_document_list():
    try:
        response = supabase.table("documents").select("metadata", count='exact').execute()
        if response.data:
            sources = [d['metadata']['source'] for d in response.data]
            df = pd.DataFrame(sources, columns=['source'])
            return df.groupby('source').size().reset_index(name='chunks')
        return pd.DataFrame(columns=['source', 'chunks'])
    except Exception as e:
        st.error(f"Error al obtener lista de documentos: {e}")
        return pd.DataFrame(columns=['source', 'chunks'])

def delete_document_by_source(source_name):
    try:
        supabase.table("documents").delete().eq("metadata->>source", source_name).execute()
        st.success(f"¡Documento '{source_name}' y todos sus chunks han sido eliminados!")
        get_document_list.clear()
    except Exception as e:
        st.error(f"Error al eliminar el documento: {e}")

def log_and_get_id(tecnico, modelo, tipo, desc, diag):
    try:
        response = supabase.table("diagnostics_log").insert({"tecnico": tecnico, "modelo_maquina": modelo, "tipo_consulta": tipo, "descripcion_averia": desc, "diagnostico_ia": diag}).select("id").execute()
        if response.data:
            st.toast("Consulta registrada en el historial.")
            get_dashboard_data.clear()
            return response.data[0]['id']
        return None
    except Exception as e:
        st.warning(f"No se pudo registrar el diagnóstico: {e}")
        return None

def update_feedback(log_id, feedback_value):
    try:
        supabase.table("diagnostics_log").update({"feedback": feedback_value}).eq("id", log_id).execute()
        st.toast("¡Gracias por tu feedback!")
        get_dashboard_data.clear()
    except Exception as e:
        st.error(f"Error al guardar feedback: {e}")

def save_verified_knowledge(query, response, verifier):
    embedding = generate_embedding(response)
    if embedding:
        try:
            supabase.table("verified_knowledge").insert({"original_query": query, "verified_response": response, "verified_by": verifier, "embedding": embedding}).execute()
            st.success("¡Respuesta verificada y guardada!")
        except Exception as e:
            st.error(f"Error al guardar conocimiento verificado: {e}")

# --- Funciones de UI y Pestañas ---
def generate_pdf_report(title, data_dict, content_text):
    if not REPORTLAB_AVAILABLE:
        st.error("La librería 'reportlab' no está instalada. No se puede generar el PDF.")
        return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(title, styles['h1']))
    story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5 * cm))
    for key, value in data_dict.items():
        story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
    story.append(Spacer(1, 1 * cm))
    cleaned_content = content_text.replace('#', '').replace('*', '')
    for paragraph in cleaned_content.split('\n'):
        if paragraph.strip():
            story.append(Paragraph(paragraph, styles['Normal']))
            story.append(Spacer(1, 0.2 * cm))
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("____________________________", styles['Normal']))
    story.append(Paragraph("Firma del Mecánico", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_work_order_pdf(case_data):
    if not REPORTLAB_AVAILABLE:
        st.error("La librería 'reportlab' no está instalada.")
        return None
    
    title = f"Informe de Cierre de Caso #{case_data['id']}"
    data_dict = {
        "Título del Caso": case_data['title'],
        "Modelo de Máquina": case_data['machine_model'],
        "Tipo de Trabajo": case_data['work_type'],
        "Prioridad": case_data['priority'],
        "Asignado a": case_data.get('assigned_to', 'N/A')
    }
    content = f"<b>Descripción Inicial:</b><br/>{case_data['description']}<br/><br/><b>Informe Final de Resolución:</b><br/>{case_data.get('final_report', 'No se ha redactado un informe.')}"
    
    return generate_pdf_report(title, data_dict, content)


def consult_tab():
    st.header("Consulta Técnica")
    if 'last_response' in st.session_state and st.button("Nueva Consulta"):
        for key in ['last_response', 'last_query_data', 'context_docs', 'verified', 'log_id', 'show_case_form']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
        
    with st.form("consulta_form"):
        modelo = st.text_input("Modelo de Máquina")
        tipo = st.selectbox("Tipo de Consulta", ["Avería", "Mantenimiento", "Recambios", "Despiece"])
        consulta = st.text_area("Descripción de la consulta", height=100)
        tecnico = st.text_input("Tu Nombre (Técnico)")
        submitted = st.form_submit_button("Buscar Solución", type="primary", use_container_width=True)

    if submitted and consulta:
        full_query = f"Modelo: {modelo or 'N/E'}\nTipo: {tipo}\n{consulta}"
        st.session_state['last_query_data'] = {'tecnico': tecnico, 'modelo': modelo, 'consulta': consulta, 'tipo': tipo}
        with st.spinner("Buscando en conocimiento verificado..."):
            verified_answer = search_verified_knowledge(full_query)
        if verified_answer:
            st.session_state['last_response'] = verified_answer['verified_response']
            st.session_state['verified'] = True
        else:
            with st.spinner("Buscando en la base de conocimiento y generando respuesta..."):
                docs = search_document_knowledge(full_query)
                respuesta_ia = generate_technical_response(full_query, docs, tipo)
                st.session_state['last_response'] = respuesta_ia
                st.session_state['context_docs'] = docs
                st.session_state['verified'] = False
        log_id = log_and_get_id(tecnico, modelo, tipo, consulta, st.session_state['last_response'])
        st.session_state['log_id'] = log_id
        st.rerun()

    if 'last_response' in st.session_state:
        st.divider()
        if st.session_state.get('verified'):
            st.success("✅ Respuesta Verificada por Dirección Técnica")
        else:
            st.info("ℹ️ Respuesta generada por IA")
        st.markdown(st.session_state['last_response'])
        
        col1, col2 = st.columns(2)
        with col1:
            pdf_buffer = generate_pdf_report("Informe de Consulta Técnica", st.session_state['last_query_data'], st.session_state['last_response'])
            if pdf_buffer:
                st.download_button(label="📥 Descargar Informe en PDF", data=pdf_buffer, file_name=f"informe_consulta_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", use_container_width=True)
        with col2:
            if st.button("📝 Crear Caso a partir de esta Consulta", use_container_width=True):
                st.session_state['show_case_form'] = True
                st.rerun()

        if not st.session_state.get('verified') and st.session_state.get('log_id') is not None:
            log_id = st.session_state['log_id']
            st.write("¿Fue útil esta respuesta?")
            cols = st.columns(10)
            cols[0].button("👍", on_click=update_feedback, args=(log_id, 1), key=f"up_{log_id}")
            cols[1].button("👎", on_click=update_feedback, args=(log_id, -1), key=f"down_{log_id}")

    if st.session_state.get('show_case_form'):
        st.divider()
        st.header("Crear Nuevo Caso de Trabajo")
        with st.form("new_case_from_consult_form"):
            q_data = st.session_state['last_query_data']
            title = st.text_input("Título del Caso", value=f"{q_data['tipo']} - {q_data['modelo']}")
            description = st.text_area("Descripción", value=q_data['consulta'], height=150)
            assigned_to = st.text_input("Asignar a Mecánico")
            priority = st.selectbox("Prioridad", ["Baja", "Media", "Alta"], index=1)
            submitted_case = st.form_submit_button("Crear Caso", type="primary")

            if submitted_case:
                create_work_order(title, description, q_data['modelo'], q_data['tipo'], assigned_to, priority, st.session_state.get('log_id'))
                st.session_state['show_case_form'] = False
                st.rerun()

def maintenance_tab():
    st.header("Generador de Planes de Mantenimiento Preventivo")
    st.markdown("Selecciona una máquina y sus horas de uso para generar un plan de mantenimiento basado en los manuales.")
    with st.form("maintenance_form"):
        modelo = st.text_input("Modelo de la Máquina")
        horas = st.number_input("Horas de Uso Actuales", min_value=1, step=10)
        submitted = st.form_submit_button("Generar Plan", use_container_width=True)

    if submitted and modelo:
        with st.spinner(f"Generando plan de mantenimiento para {modelo} con {horas}h..."):
            plan = generate_maintenance_plan(modelo, horas)
            st.session_state['maintenance_plan'] = plan
            st.session_state['maintenance_data'] = {'modelo': modelo, 'horas_de_uso': horas}

    if 'maintenance_plan' in st.session_state:
        st.divider()
        st.subheader("Plan de Mantenimiento Sugerido")
        st.markdown(st.session_state['maintenance_plan'])
        pdf_buffer = generate_pdf_report("Plan de Mantenimiento Preventivo", st.session_state['maintenance_data'], st.session_state['maintenance_plan'])
        if pdf_buffer:
            st.download_button(label="📥 Descargar Plan en PDF", data=pdf_buffer, file_name=f"plan_mantenimiento_{st.session_state['maintenance_data']['modelo']}.pdf", mime="application/pdf")

def calculator_tab():
    st.header("Calculadora de Estimaciones")
    st.markdown("Obtén una estimación de tiempo y coste para un trabajo técnico.")
    with st.form("calculator_form"):
        modelo_calc = st.text_input("Modelo de la Máquina")
        tipo_trabajo = st.selectbox("Tipo de Trabajo", ["Reparación", "Mantenimiento"])
        desc_trabajo = st.text_area("Descripción del Trabajo a Realizar", height=100)
        tarifa_hora = st.number_input("Tarifa por Hora del Mecánico (€)", min_value=1.0, value=45.0, step=0.5)
        submitted_calc = st.form_submit_button("Calcular Estimación", use_container_width=True)

    if submitted_calc and desc_trabajo:
        with st.spinner("Generando estimación con IA..."):
            estimacion = generate_budget_estimate(tipo_trabajo, modelo_calc, desc_trabajo)
            st.session_state['last_estimation'] = estimacion
            st.session_state['last_rate'] = tarifa_hora

    if 'last_estimation' in st.session_state and st.session_state['last_estimation']:
        est = st.session_state['last_estimation']
        rate = st.session_state['last_rate']
        st.divider()
        st.subheader("Resultados de la Estimación")
        tiempo = est.get('tiempo_horas', 0)
        coste_mano_obra = tiempo * rate
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Tiempo Estimado", f"{tiempo:.1f} horas")
        col2.metric("Coste Mano de Obra", f"{coste_mano_obra:.2f} €")
        col3.metric("Dificultad", est.get('dificultad', 'N/A'))
        
        with st.expander("Justificación de la estimación"):
            st.write(est.get('justificacion_tiempo', 'Sin justificación.'))

def dashboard_tab():
    st.header("Dashboard de Inteligencia Técnica")
    st.markdown("Analiza el uso y la efectividad del asistente técnico en tiempo real.")
    df = get_dashboard_data()
    if df.empty:
        st.info("No hay datos suficientes para generar el dashboard. Realiza algunas consultas primero.")
        return
    st.subheader("Indicadores Clave de Rendimiento (KPIs)")
    seven_days_ago = pd.Timestamp.now(tz='Europe/Madrid') - pd.Timedelta(days=7)
    df_last_7_days = df[df['created_at'] >= seven_days_ago]
    total_queries = len(df)
    queries_last_7_days = len(df_last_7_days)
    feedback_counts = df['feedback'].value_counts()
    util = feedback_counts.get(1, 0)
    no_util = feedback_counts.get(-1, 0)
    total_feedback = util + no_util
    satisfaction = (util / total_feedback * 100) if total_feedback > 0 else 0
    queries_no_feedback = total_queries - total_feedback
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Consultas", total_queries)
    col2.metric("Consultas (Últ. 7 Días)", queries_last_7_days)
    col3.metric("Índice de Satisfacción", f"{satisfaction:.1f}%", help="Porcentaje de respuestas marcadas como 'útiles' sobre el total de respuestas valoradas.")
    col4.metric("Consultas sin Valorar", queries_no_feedback)
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Rendimiento de la IA")
        if total_feedback > 0:
            feedback_df = pd.DataFrame({'Valoración': ['Útil', 'No Útil', 'Sin Valorar'], 'Cantidad': [util, no_util, queries_no_feedback]})
            st.bar_chart(feedback_df.set_index('Valoración'))
        else:
            st.info("Aún no se ha valorado ninguna respuesta.")
        st.subheader("Consultas por Técnico")
        df['tecnico'] = df['tecnico'].fillna('Anónimo')
        st.bar_chart(df['tecnico'].value_counts().head(10))
    with col_b:
        st.subheader("Máquinas Más Consultadas")
        st.bar_chart(df['modelo_maquina'].value_counts().head(10))
        st.subheader("Tipos de Consulta Frecuentes")
        st.bar_chart(df['tipo_consulta'].value_counts())

def history_tab():
    st.header("Historial y Verificación de Consultas")
    try:
        logs = supabase.table("diagnostics_log").select("*").order("created_at", desc=True).limit(20).execute().data
        if logs:
            for log in logs:
                with st.expander(f"{log['created_at'][:10]} - {log['modelo_maquina']} - {log['descripcion_averia'][:50]}"):
                    st.markdown(f"**Consulta:** {log['descripcion_averia']}")
                    st.markdown(f"**Respuesta IA:** {log['diagnostico_ia']}")
                    if st.button("✅ Validar y Guardar", key=f"verify_{log['id']}"):
                        st.session_state['verify_item'] = log
    except Exception as e:
        st.error(f"Error al cargar historial: {e}")
    if 'verify_item' in st.session_state:
        log_to_verify = st.session_state['verify_item']
        st.subheader("Verificar y Guardar Respuesta")
        verified_response = st.text_area("Edita la respuesta para guardarla como oficial:", value=log_to_verify['diagnostico_ia'], height=200, key=f"text_{log_to_verify['id']}")
        verifier = st.text_input("Tu nombre (verificador):", key=f"verifier_{log_to_verify['id']}")
        if st.button("Guardar Conocimiento Verificado", key=f"save_{log_to_verify['id']}"):
            if verifier:
                save_verified_knowledge(log_to_verify['descripcion_averia'], verified_response, verifier)
                del st.session_state['verify_item']
                st.rerun()
            else:
                st.warning("Por favor, introduce tu nombre como verificador.")

def knowledge_management_tab():
    st.header("Gestión de la Base de Conocimiento")
    st.markdown("Aquí puedes ver y eliminar los documentos que forman la memoria del asistente.")
    docs_df = get_document_list()
    if not docs_df.empty:
        st.dataframe(docs_df, use_container_width=True)
        doc_to_delete = st.selectbox("Selecciona un documento para eliminar:", options=docs_df['source'].tolist())
        if st.button("Eliminar Documento Seleccionado", type="primary"):
            if doc_to_delete:
                delete_document_by_source(doc_to_delete)
                st.rerun()
    else:
        st.info("No hay documentos en la base de conocimiento. Sube algunos desde la barra lateral.")

def cmms_tab():
    st.header("Gestión de Casos (CMMS)")
    st.markdown("Crea y gestiona órdenes de trabajo para averías y mantenimientos preventivos.")

    if st.button("➕ Crear Nuevo Caso"):
        st.session_state['show_new_case_form'] = True
        if 'case_to_close' in st.session_state:
            del st.session_state['case_to_close']
    
    if st.session_state.get('show_new_case_form'):
        with st.form("new_case_form"):
            st.subheader("Detalles del Nuevo Caso")
            title = st.text_input("Título del Caso", placeholder="Ej: Mantenimiento 500h John Deere 6110M")
            machine_model = st.text_input("Modelo de Máquina")
            description = st.text_area("Descripción Detallada")
            c1, c2, c3 = st.columns(3)
            work_type = c1.selectbox("Tipo de Trabajo", ["Avería", "Preventivo", "Mejora", "Inspección"])
            assigned_to = c2.text_input("Asignar a")
            priority = c3.selectbox("Prioridad", ["Baja", "Media", "Alta"], index=1)
            submitted = st.form_submit_button("Guardar Caso", type="primary")

            if submitted:
                if title and machine_model:
                    create_work_order(title, description, machine_model, work_type, assigned_to, priority)
                    st.session_state['show_new_case_form'] = False
                    st.rerun()
                else:
                    st.warning("El título y el modelo de la máquina son obligatorios.")

    if 'case_to_close' in st.session_state:
        case = st.session_state['case_to_close']
        st.subheader(f"Finalizar y Redactar Informe del Caso #{case['id']}")
        with st.form(key=f"close_form_{case['id']}"):
            report_text = st.text_area("Informe de Resolución:", height=200, help="Detalla los trabajos realizados, las piezas cambiadas y cualquier observación relevante.")
            submit_report = st.form_submit_button("Finalizar Caso y Guardar Informe")

            if submit_report:
                if report_text:
                    update_work_order_with_report(case['id'], report_text)
                    del st.session_state['case_to_close']
                    st.rerun()
                else:
                    st.warning("El informe de resolución es obligatorio para cerrar un caso.")

    st.divider()
    st.subheader("Tablero de Casos")
    
    work_orders = get_work_orders()
    if not work_orders:
        st.info("No hay casos de trabajo activos. ¡Crea uno para empezar!")
        return

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='kanban-header'><h3>Abierto</h3></div>", unsafe_allow_html=True)
        for case in [wo for wo in work_orders if wo['status'] == 'Abierto']:
            with st.container(border=True):
                st.markdown(f"**{case['title']}**")
                st.caption(f"#{case['id']} | Prioridad: {case['priority']}")
                st.markdown(f"**Máquina:** {case['machine_model']}")
                if case.get('assigned_to'):
                    st.markdown(f"**Asignado a:** {case['assigned_to']}")
                if st.button("▶️ Empezar", key=f"start_{case['id']}", use_container_width=True):
                    update_work_order_status(case['id'], 'En Progreso')
                    st.rerun()

    with col2:
        st.markdown("<div class='kanban-header'><h3>En Progreso</h3></div>", unsafe_allow_html=True)
        for case in [wo for wo in work_orders if wo['status'] == 'En Progreso']:
            with st.container(border=True):
                st.markdown(f"**{case['title']}**")
                st.caption(f"#{case['id']} | Prioridad: {case['priority']}")
                st.markdown(f"**Máquina:** {case['machine_model']}")
                if case.get('assigned_to'):
                    st.markdown(f"**Asignado a:** {case['assigned_to']}")
                if st.button("✅ Finalizar", key=f"finish_{case['id']}", use_container_width=True):
                    st.session_state['case_to_close'] = case
                    if 'show_new_case_form' in st.session_state:
                        del st.session_state['show_new_case_form']
                    st.rerun()
    
    with col3:
        st.markdown("<div class='kanban-header'><h3>Cerrado</h3></div>", unsafe_allow_html=True)
        for case in [wo for wo in work_orders if wo['status'] == 'Cerrado']:
            with st.container(border=True):
                st.markdown(f"**{case['title']}**")
                st.caption(f"#{case['id']} | Prioridad: {case['priority']}")
                with st.expander("Ver Informe Final"):
                    st.markdown(case.get('final_report', 'No hay informe disponible.'))
                
                pdf_buffer = generate_work_order_pdf(case)
                if pdf_buffer:
                    st.download_button(
                        label="📥 Descargar Informe",
                        data=pdf_buffer,
                        file_name=f"informe_caso_{case['id']}_{case['machine_model']}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{case['id']}"
                    )

# --- Navegación Principal y Renderizado de Páginas ---

def render_hub_page():
    st.title("🛠️ Asistente Técnico Satgarden V2.8")
    st.markdown("""
    **Bienvenido al centro de operaciones técnicas de Satgarden.** Esta plataforma es tu copiloto para la gestión del conocimiento, diagnósticos y operaciones de mantenimiento.
    
    Selecciona una herramienta para empezar:
    """)
    st.divider()

    # We use markdown with unsafe_allow_html to create the button text with an icon
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("<span>💬</span>Consulta Técnica", use_container_width=True):
            st.session_state.page = "Consulta"
            st.rerun()
        if st.button("<span>📊</span>Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()
    with col2:
        if st.button("<span>📋</span>Gestión de Casos", use_container_width=True):
            st.session_state.page = "CMMS"
            st.rerun()
        if st.button("<span>⚙️</span>Mantenimiento Preventivo", use_container_width=True):
            st.session_state.page = "Mantenimiento"
            st.rerun()
    with col3:
        if st.button("<span>🧮</span>Calculadora", use_container_width=True):
            st.session_state.page = "Calculadora"
            st.rerun()
        if st.button("<span>📚</span>Base de Conocimiento", use_container_width=True):
            st.session_state.page = "Conocimiento"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("<span>📜</span>Ver Historial y Verificar", use_container_width=True):
        st.session_state.page = "Historial"
        st.rerun()


def render_full_app():
    if st.button("⬅️ Volver al Menú Principal"):
        st.session_state.page = "Hub"
        # Clear specific states when going back to hub to avoid stale data
        keys_to_clear = ['last_response', 'maintenance_plan', 'last_estimation', 'case_to_close', 'show_case_form', 'verify_item']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    page_map = {
        "Consulta": consult_tab,
        "CMMS": cmms_tab,
        "Mantenimiento": maintenance_tab,
        "Calculadora": calculator_tab,
        "Dashboard": dashboard_tab,
        "Historial": history_tab,
        "Conocimiento": knowledge_management_tab,
    }
    
    page_function = page_map.get(st.session_state.page)
    if page_function:
        page_function()
    else:
        st.error("Página no encontrada.")
        st.session_state.page = "Hub"
        st.rerun()


def main():
    load_css()

    with st.sidebar:
        try:
            st.image("logo.png", use_container_width=True)
        except Exception:
            pass # Fails silently if logo is not found
        st.header("Administración")
        with st.expander("Cargar Documentos", expanded=True):
            uploaded_files = st.file_uploader("Sube manuales en formato PDF", type=['pdf'], accept_multiple_files=True)
            if st.button("Procesar y Guardar PDFs"):
                if uploaded_files:
                    ingest_pdf_files(uploaded_files)
                else:
                    st.warning("Por favor, selecciona al menos un archivo PDF.")
    
    if 'page' not in st.session_state:
        st.session_state.page = "Hub"

    if st.session_state.page == "Hub":
        render_hub_page()
    else:
        render_full_app()

if __name__ == "__main__":
    main()

