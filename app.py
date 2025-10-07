"""
ASISTENTE TÉCNICO SATGARDEN V1.3
Iteración con mejoras de rendimiento (caché), gestión de documentos (eliminar),
exportación a PDF funcional y mejoras en la interfaz de usuario.
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
from io import BytesIO

# --- Dependencia Opcional: Reportlab para PDFs ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- Inicialización y Configuración ---
load_dotenv()

# Configuración de página de Streamlit (debe ser el primer comando de st)
st.set_page_config(page_title="Asistente Satgarden", page_icon="🔧", layout="wide")

# --- Gestión de Caché y Conexiones ---

@st.cache_resource
def get_openai_client():
    """Crea y cachea el cliente de OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

@st.cache_resource
def get_supabase_client():
    """Crea y cachea el cliente de Supabase."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not (supabase_url and supabase_key):
        st.error("Faltan las variables de entorno de Supabase. Revisa tu archivo .env.")
        st.stop()
    client = create_client(supabase_url, supabase_key)
    return client

# Inicialización de Clientes
try:
    openai_client = get_openai_client()
    if not openai_client.api_key:
        st.error("Falta la clave de API de OpenAI. Revisa tu archivo .env.")
        st.stop()
    supabase = get_supabase_client()
except Exception as e:
    st.error(f"Error al inicializar los clientes: {e}")
    st.stop()

# --- Constantes ---
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Funciones de Procesamiento de Datos ---

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                clean_text = re.sub(r'\s+', ' ', page_text.replace('\x00', ''))
                text += clean_text + "\n\n"
    except Exception as e:
        st.error(f"Error al leer el PDF: {e}")
    return text.strip()

def chunk_text(text, chunk_size=2000, chunk_overlap=200):
    if not isinstance(text, str) or len(text) < 100: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return [c for c in chunks if c.strip()]

# --- Funciones de IA (OpenAI) ---

def generate_embedding(text):
    try:
        text = text.replace("\n", " ").strip()[:8191]
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"No se pudo generar el embedding: {e}")
        return None

def generate_technical_response(query, context_docs, tipo_consulta):
    context = "\n\n---\n\n".join([f"Fuente: {doc.get('metadata', {}).get('source', 'Desconocida')}\nContenido: {doc['content']}" for doc in context_docs]) if context_docs else "No se encontró información de contexto relevante."
    system_prompts = {
        "Mantenimiento": "Eres un técnico experto de Satgarden. Proporciona un plan de mantenimiento estructurado, claro y paso a paso. Usa listas numeradas.",
        "Avería": "Eres un técnico de diagnóstico de Satgarden. Analiza la avería. Proporciona una lista de posibles causas por orden de probabilidad y los pasos para diagnosticarlas."
    }
    system_prompt = system_prompts.get(tipo_consulta, "Eres un asistente técnico de Satgarden. Responde de forma clara y concisa.")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Basándote en el CONTEXTO, responde a la CONSULTA.\n\nCONTEXTO:\n{context}\n\nCONSULTA:\n{query}"}
            ],
            temperature=0.2, max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al generar la respuesta: {e}"

def generate_budget_estimate(query, context_docs):
    context = "\n\n---\n\n".join([f"Fuente: {doc.get('metadata', {}).get('source', 'Desconocida')}\nContenido: {doc['content']}" for doc in context_docs]) if context_docs else "No se encontró información de contexto relevante."
    system_prompt = """
    Eres un jefe de taller experto en maquinaria de Satgarden. Tu tarea es estimar el tiempo necesario para trabajos técnicos.
    Analiza el contexto y la descripción de la tarea. Responde ÚNICAMENTE con un objeto JSON con el formato:
    {"tiempo_horas": float, "justificacion": "string con tu razonamiento", "dificultad": "Baja/Media/Alta"}
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXTO:\n{context}\n\nTAREA A ESTIMAR:\n{query}"}
            ],
            temperature=0.1, max_tokens=500, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error al generar la estimación: {e}")
        return None

# --- Funciones de Gestión de Datos (Supabase) ---

def store_document_chunk(content, metadata):
    embedding = generate_embedding(content)
    if embedding:
        try:
            supabase.table("documents").insert({
                "content": content, "metadata": metadata, "embedding": embedding
            }).execute()
            return True
        except Exception as e:
            st.error(f"Error al guardar en Supabase: {e}")
    return False

def ingest_pdf_files(pdf_files):
    for pdf in pdf_files:
        st.info(f"Procesando: {pdf.name}")
        text = extract_text_from_pdf(pdf)
        if not text:
            st.warning(f"No se pudo extraer texto de {pdf.name}.")
            continue
        chunks = chunk_text(text)
        progress_bar = st.progress(0, text=f"Guardando chunks de {pdf.name}")
        success_count = sum(1 for i, chunk in enumerate(chunks) if store_document_chunk(chunk, {"source": pdf.name, "chunk_index": i}))
        progress_bar.progress(1.0)
        st.success(f"Se guardaron {success_count} chunks de {pdf.name}")
    st.cache_data.clear()

def search_similar_documents(query_text, top_k=5):
    embedding = generate_embedding(query_text)
    if not embedding: return []
    try:
        result = supabase.rpc('match_documents', {
            'query_embedding': embedding, 'match_count': top_k
        }).execute()
        return result.data if result.data else []
    except Exception:
        st.error("Error en la búsqueda. Asegúrate de que la función `match_documents` existe en Supabase.")
        return []

def log_diagnostic(tecnico, modelo, desc, diag):
    try:
        supabase.table("diagnostics_log").insert({
            "tecnico": tecnico, "modelo_maquina": modelo, "descripcion_averia": desc, "diagnostico_ia": diag
        }).execute()
    except Exception as e:
        st.warning(f"No se pudo registrar el diagnóstico: {e}")

@st.cache_data(ttl=3600) # Cache por 1 hora
def get_knowledge_base_summary():
    try:
        response = supabase.table("documents").select("metadata", count="exact").execute()
        if response.data:
            sources = [item['metadata']['source'] for item in response.data if 'metadata' in item and 'source' in item['metadata']]
            if sources:
                df_counts = pd.Series(sources).value_counts().reset_index()
                df_counts.columns = ['Documento', 'Nº de Fragmentos']
                return df_counts
    except Exception as e:
        st.error(f"Error al consultar la base de conocimiento: {e}")
    return pd.DataFrame(columns=['Documento', 'Nº de Fragmentos'])

def delete_document_by_source(source_name):
    try:
        supabase.table("documents").delete().eq("metadata->>source", source_name).execute()
        st.success(f"Documento '{source_name}' eliminado correctamente.")
        st.cache_data.clear() # Limpiar caché para refrescar la lista
    except Exception as e:
        st.error(f"Error al eliminar el documento: {e}")

# --- Funciones de Generación de Informes ---

def generate_pdf_report(query_data, response_text, sources):
    if not REPORTLAB_AVAILABLE:
        st.error("La librería Reportlab no está instalada. No se puede generar el PDF.")
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))

    story = []
    story.append(Paragraph("<b>INFORME TÉCNICO - SATGARDEN</b>", styles['h1']))
    story.append(Paragraph(f"<i>Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>", styles['Center']))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph(f"<b>Técnico:</b> {query_data.get('tecnico', 'No especificado')}", styles['Normal']))
    story.append(Paragraph(f"<b>Máquina:</b> {query_data.get('modelo', 'No especificada')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("<u>Consulta Realizada:</u>", styles['h3']))
    story.append(Paragraph(query_data.get('consulta', ''), styles['Justify']))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("<u>Respuesta del Asistente de IA:</u>", styles['h3']))
    # Limpiar un poco el markdown para el PDF
    clean_response = response_text.replace("*", "").replace("#", "")
    for line in clean_response.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Justify']))
            story.append(Spacer(1, 0.1*cm))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Pestañas de la Interfaz de Usuario (UI) ---

def main_consult_tab():
    st.header("Consulta Técnica")
    if 'last_response' in st.session_state and st.button("Iniciar Nueva Consulta"):
        del st.session_state['last_response']
        del st.session_state['context_docs']
        st.rerun()

    with st.form("consulta_form"):
        modelo = st.text_input("Modelo de la Máquina", placeholder="Ej: Tractor ZT50")
        tipo = st.selectbox("Tipo de Consulta", ["Avería", "Mantenimiento", "Recambios", "Despiece"])
        consulta = st.text_area("Describe tu consulta", height=120, placeholder="Ej: El motor de arranque no funciona...")
        tecnico = st.text_input("Tu Nombre (Técnico)", placeholder="Para el registro y el informe")
        submitted = st.form_submit_button("Buscar Solución", type="primary", use_container_width=True)

    if submitted and consulta:
        query = f"Modelo: {modelo or 'N/E'}\nTipo: {tipo}\n{consulta}"
        with st.spinner("Buscando en la base de conocimiento..."):
            docs = search_similar_documents(query)
        with st.spinner("Generando respuesta con el asistente de IA..."):
            respuesta = generate_technical_response(query, docs, tipo)
            log_diagnostic(tecnico or "Anónimo", modelo or "N/E", consulta, respuesta)
        st.session_state['last_response'] = respuesta
        st.session_state['last_query_data'] = {'tecnico': tecnico, 'modelo': modelo, 'consulta': consulta}
        st.session_state['context_docs'] = docs
        st.rerun()

    if 'last_response' in st.session_state:
        st.divider()
        st.subheader("Respuesta del Asistente")
        st.markdown(st.session_state['last_response'])
        pdf_buffer = generate_pdf_report(st.session_state['last_query_data'], st.session_state['last_response'], st.session_state['context_docs'])
        if pdf_buffer:
            st.download_button(label="Descargar Informe PDF", data=pdf_buffer, file_name=f"informe_satgarden_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
        
        with st.expander("Ver fuentes consultadas"):
            docs = st.session_state.get('context_docs', [])
            if docs:
                for doc in docs:
                    st.info(f"**Fuente:** {doc.get('metadata', {}).get('source', 'N/A')} (Similitud: {doc.get('similarity', 0):.2f})")
            else:
                st.write("No se consultaron fuentes específicas.")

def calculator_tab():
    st.header("Calculadora de Tiempos y Costes")
    with st.form("calculator_form"):
        modelo = st.text_input("Modelo de la Máquina", placeholder="Ej: Cosechadora X-100")
        tipo_trabajo = st.selectbox("Tipo de Trabajo", ["Mantenimiento", "Reparación"])
        desc_trabajo = st.text_area("Descripción del trabajo", height=120, placeholder="Ej: Mantenimiento de las 500 horas...")
        tarifa = st.number_input("Tarifa del mecánico (€/hora)", min_value=20.0, value=45.0, step=1.0)
        submitted = st.form_submit_button("Calcular Estimación", type="primary", use_container_width=True)

    if submitted and desc_trabajo:
        query = f"Modelo: {modelo or 'N/E'}\nTipo: {tipo_trabajo}\n{desc_trabajo}"
        with st.spinner("Calculando estimación..."):
            docs = search_similar_documents(query, top_k=4)
            estimacion = generate_budget_estimate(query, docs)
        if estimacion:
            st.session_state['last_estimate'] = estimacion
            st.session_state['last_tarifa'] = tarifa

    if 'last_estimate' in st.session_state:
        st.divider()
        st.subheader("Estimación Generada")
        est = st.session_state['last_estimate']
        coste = est.get('tiempo_horas', 0.0) * st.session_state['last_tarifa']
        col1, col2, col3 = st.columns(3)
        col1.metric("Tiempo Estimado", f"{est.get('tiempo_horas', 0.0):.1f} horas")
        col2.metric("Coste Mano de Obra", f"{coste:.2f} €")
        col3.metric("Dificultad", est.get('dificultad', 'N/A'))
        with st.expander("Ver justificación de la IA"):
            st.info(est.get('justificacion', 'Sin justificación.'))

def history_tab():
    st.header("Historial de Consultas")
    try:
        logs = supabase.table("diagnostics_log").select("*").order("created_at", desc=True).limit(50).execute().data
        if logs:
            df = pd.DataFrame(logs)[['created_at', 'tecnico', 'modelo_maquina', 'descripcion_averia']]
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error al cargar el historial: {e}")

def knowledge_base_tab():
    st.header("Base de Conocimiento")
    st.info("Visualiza y gestiona los documentos en la memoria del asistente.")
    
    df_summary = get_knowledge_base_summary()
    
    if not df_summary.empty:
        # Crear una versión del dataframe con botones
        st.write("Documentos cargados:")
        for index, row in df_summary.iterrows():
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(row['Documento'])
            with col2:
                st.write(f"{row['Nº de Fragmentos']} fragmentos")
            with col3:
                if st.button("Eliminar", key=f"delete_{row['Documento']}"):
                    delete_document_by_source(row['Documento'])
                    st.rerun()
    else:
        st.warning("La base de conocimiento está vacía.")

# --- Aplicación Principal ---

def main():
    st.title("🔧 Asistente Técnico Satgarden V1.3")
    
    with st.sidebar:
        st.header("Administración")
        try:
            count = supabase.table("documents").select("id", count="exact").execute().count
        except Exception:
            count = "Error"
        st.metric("Chunks en BBDD", count)

        with st.expander("Cargar Manuales (PDF)"):
            pdfs = st.file_uploader("Sube archivos PDF", type=['pdf'], accept_multiple_files=True)
            if st.button("Procesar y Guardar PDFs"):
                if pdfs:
                    ingest_pdf_files(pdfs)
                else:
                    st.warning("Selecciona al menos un archivo PDF.")
    
    tabs = st.tabs(["Consulta Técnica", "Calculadora", "Historial", "Base de Conocimiento"])
    with tabs[0]:
        main_consult_tab()
    with tabs[1]:
        calculator_tab()
    with tabs[2]:
        history_tab()
    with tabs[3]:
        knowledge_base_tab()

if __name__ == "__main__":
    main()
