"""
ASISTENTE TÉCNICO SATGARDEN V1.2
Versión con pestaña "Base de Conocimiento" para visualizar los documentos cargados.
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

# --- Dependencia Opcional: Reportlab para PDFs ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.sidebar.warning("Reportlab no está instalado. La generación de PDFs está desactivada. Instálalo con: pip install reportlab")

# --- Inicialización y Configuración ---
load_dotenv()

# Configuración de página de Streamlit (debe ser el primer comando de st)
st.set_page_config(page_title="Asistente Satgarden", page_icon="🔧", layout="wide")

# Inicialización de Clientes (OpenAI y Supabase)
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not (openai_client.api_key and supabase_url and supabase_key):
        st.error("Faltan las variables de entorno. Asegúrate de tener OPENAI_API_KEY, SUPABASE_URL y SUPABASE_KEY en tu archivo .env")
        st.stop()
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error(f"Error al inicializar los clientes: {e}")
    st.stop()

# --- Constantes ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536 # Dimensión para text-embedding-3-small

# --- Funciones de Backend (Procesamiento de Datos) ---

def extract_text_from_pdf(pdf_file):
    """Extrae texto de un objeto de archivo PDF subido."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Limpieza básica del texto extraído
                clean_text = re.sub(r'\s+', ' ', page_text.replace('\x00', ''))
                text += clean_text + "\n\n"
    except Exception as e:
        st.error(f"Error al leer el PDF: {e}")
    return text.strip()

def chunk_text(text, chunk_size=2000, chunk_overlap=200):
    """Divide el texto en chunks con superposición."""
    if not isinstance(text, str) or len(text) < 100:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return [c for c in chunks if c.strip()]

def generate_embedding(text):
    """Genera un embedding para un texto dado."""
    try:
        # Asegura que el texto no exceda el límite del modelo
        text = text.replace("\n", " ").strip()[:8191]
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"No se pudo generar el embedding: {e}")
        return None

def store_document_chunk(content, metadata):
    """Genera embedding y almacena un chunk en Supabase."""
    try:
        embedding = generate_embedding(content)
        if embedding:
            # Verifica que la dimensión del embedding sea la correcta
            if len(embedding) != EMBEDDING_DIMENSION:
                st.error(f"Error de dimensión del embedding. Esperado: {EMBEDDING_DIMENSION}, Obtenido: {len(embedding)}")
                return None
            
            response = supabase.table("documents").insert({
                "content": content,
                "metadata": metadata,
                "embedding": embedding
            }).execute()
            return response
    except Exception as e:
        st.error(f"Error al guardar en Supabase: {e}")
    return None

def ingest_pdf_files(pdf_files, doc_type="manual"):
    """Procesa una lista de archivos PDF subidos."""
    total_chunks_saved = 0
    for pdf in pdf_files:
        st.info(f"Procesando: {pdf.name}")
        text = extract_text_from_pdf(pdf)
        if not text:
            st.warning(f"No se pudo extraer texto de {pdf.name}. Saltando archivo.")
            continue
        
        chunks = chunk_text(text)
        if not chunks:
            st.warning(f"No se generaron chunks de texto para {pdf.name}.")
            continue
        
        progress_bar = st.progress(0, text=f"Guardando chunks de {pdf.name}")
        success_count = 0
        for i, chunk in enumerate(chunks):
            metadata = {"source": pdf.name, "type": doc_type, "chunk_index": i}
            if store_document_chunk(chunk, metadata):
                success_count += 1
            progress_bar.progress((i + 1) / len(chunks), text=f"Guardando chunks de {pdf.name}")
        
        if success_count > 0:
            st.success(f"Se guardaron {success_count} chunks de {pdf.name}")
            total_chunks_saved += success_count
        else:
            st.error(f"No se pudo guardar ningún chunk de {pdf.name}")
    return total_chunks_saved

# --- Funciones de Lógica de la Aplicación ---

def search_similar_documents(query_text, top_k=5):
    """Busca documentos similares usando la función RPC de Supabase."""
    try:
        embedding = generate_embedding(query_text)
        if not embedding:
            return []
        
        result = supabase.rpc('match_documents', {
            'query_embedding': embedding,
            'match_count': top_k
        }).execute()
        
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Error en la búsqueda de documentos: {e}")
        st.info("Asegúrate de haber creado la función `match_documents` en el SQL Editor de Supabase como se indica en el README.")
        return []

def generate_technical_response(query, context_docs, tipo_consulta):
    """Genera una respuesta técnica usando el LLM con contexto."""
    context = "\n\n---\n\n".join([f"Fuente: {doc.get('metadata', {}).get('source', 'Desconocida')}\nContenido: {doc['content']}" for doc in context_docs]) if context_docs else "No se encontró información de contexto relevante."
    
    system_prompts = {
        "Mantenimiento": "Eres un técnico experto de Satgarden. Proporciona un plan de mantenimiento estructurado, claro y paso a paso. Usa listas numeradas.",
        "Recambios": "Eres un especialista en recambios de Satgarden. Identifica los recambios necesarios, sus códigos si es posible, y dónde encontrarlos en la máquina.",
        "Despiece": "Eres un técnico de Satgarden. Describe los componentes principales de la máquina, su ubicación y su función, basándote en el contexto.",
        "Avería": "Eres un técnico de diagnóstico de Satgarden. Analiza la avería descrita. Proporciona una lista de posibles causas por orden de probabilidad y los pasos para diagnosticarlas."
    }
    system_prompt = system_prompts.get(tipo_consulta, "Eres un asistente técnico de Satgarden. Responde de forma clara y concisa.")

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Basándote en el siguiente CONTEXTO, responde a la CONSULTA del usuario.\n\nCONTEXTO:\n{context}\n\nCONSULTA:\n{query}"}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al generar la respuesta: {e}"

def generate_budget_estimate(query, context_docs):
    """Genera una estimación de tiempo para un trabajo técnico usando el LLM."""
    context = "\n\n---\n\n".join([f"Fuente: {doc.get('metadata', {}).get('source', 'Desconocida')}\nContenido: {doc['content']}" for doc in context_docs]) if context_docs else "No se encontró información de contexto relevante."

    system_prompt = """
    Eres un jefe de taller experto en maquinaria de Satgarden. Tu tarea es estimar el tiempo necesario para realizar trabajos técnicos.
    Analiza el contexto de trabajos similares y la descripción de la nueva tarea.
    Responde ÚNICAMENTE con un objeto JSON con el siguiente formato:
    {"tiempo_horas": float, "justificacion": "string con tu razonamiento", "dificultad": "Baja/Media/Alta"}
    No añadas texto antes o después del JSON.
    """

    user_message = f"""
    Basándote en el siguiente CONTEXTO de trabajos y manuales, estima el tiempo para la TAREA descrita.

    CONTEXTO:
    {context}

    TAREA A ESTIMAR:
    {query}
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar la estimación: {e}")
        return None

def extract_json_from_response(text):
    """Extrae un bloque JSON de una respuesta de texto, incluso si está dentro de ```json ... ```."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    else:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            st.error("La respuesta del modelo no es un JSON válido.")
            return None

def log_diagnostic(tecnico, modelo, desc, diag):
    """Registra una consulta en la tabla de logs."""
    try:
        supabase.table("diagnostics_log").insert({
            "tecnico": tecnico,
            "modelo_maquina": modelo,
            "descripcion_averia": desc,
            "diagnostico_ia": diag
        }).execute()
    except Exception as e:
        st.warning(f"No se pudo registrar el diagnóstico: {e}")

# --- Funciones de Interfaz de Usuario (UI) ---

def main_consult_tab():
    st.header("Consulta Técnica")
    
    with st.form("consulta_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            modelo = st.text_input("Modelo de la Máquina", key="modelo_consulta", placeholder="Ej: Tractor ZT50")
        with col2:
            tipo = st.selectbox("Tipo de Consulta", ["Avería", "Mantenimiento", "Recambios", "Despiece"], key="tipo_consulta")
        
        consulta = st.text_area("Describe tu consulta", height=120, key="consulta_texto", placeholder="Ej: El motor de arranque no funciona, hace un clic pero no gira.")
        tecnico = st.text_input("Tu Nombre (Técnico)", key="tecnico_nombre", placeholder="Para el registro y el informe")
        
        submitted = st.form_submit_button("Buscar Solución", type="primary", use_container_width=True)

    if submitted:
        if not consulta:
            st.warning("Por favor, introduce una consulta.")
        else:
            full_query = f"Modelo: {modelo or 'No especificado'}\nTipo: {tipo}\nConsulta: {consulta}"
            
            with st.spinner("Buscando en la base de conocimiento..."):
                docs = search_similar_documents(full_query)
            
            with st.spinner("Generando respuesta con el asistente de IA..."):
                respuesta = generate_technical_response(full_query, docs, tipo)
                log_diagnostic(tecnico or "Anónimo", modelo or "No especificado", consulta, respuesta)

            st.session_state['last_response'] = respuesta
            st.session_state['last_query_data'] = {
                'tecnico': tecnico, 'modelo': modelo, 'consulta': consulta
            }
            st.session_state['context_docs'] = docs

    if 'last_response' in st.session_state:
        st.divider()
        st.subheader("Respuesta del Asistente")
        st.markdown(st.session_state['last_response'])

        if st.session_state.get('context_docs'):
            with st.expander("Ver fuentes consultadas"):
                for doc in st.session_state['context_docs']:
                    st.info(f"**Fuente:** {doc.get('metadata', {}).get('source', 'N/A')} (Similitud: {doc.get('similarity', 0):.2f})")
                    st.caption(doc['content'])
        
        if REPORTLAB_AVAILABLE:
            pass

def calculator_tab():
    st.header("Calculadora de Tiempos y Costes")
    st.caption("Estima el tiempo y coste para mantenimientos o reparaciones basado en la base de conocimiento.")

    with st.form("calculator_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            modelo = st.text_input("Modelo de la Máquina", placeholder="Ej: Cosechadora X-100")
        with col2:
            tipo_trabajo = st.selectbox("Tipo de Trabajo", ["Mantenimiento", "Reparación"])

        desc_trabajo = st.text_area("Descripción detallada del trabajo a realizar", height=120, placeholder="Ej: Realizar mantenimiento de las 500 horas, incluyendo cambio de aceite de motor, filtro de aire y revisión de correas.")
        tarifa = st.number_input("Tarifa del mecánico (€/hora)", min_value=20.0, max_value=150.0, value=45.0, step=1.0)

        submitted = st.form_submit_button("Calcular Estimación", type="primary", use_container_width=True)

    if submitted:
        if not desc_trabajo:
            st.warning("Por favor, introduce una descripción del trabajo.")
        else:
            full_query = f"Modelo: {modelo or 'No especificado'}\nTipo: {tipo_trabajo}\nDescripción: {desc_trabajo}"

            with st.spinner("Buscando trabajos similares en la base de datos..."):
                docs = search_similar_documents(full_query, top_k=4)

            with st.spinner("La IA está calculando la estimación..."):
                estimacion_json_str = generate_budget_estimate(full_query, docs)

            if estimacion_json_str:
                estimacion_obj = extract_json_from_response(estimacion_json_str)
                if estimacion_obj:
                    st.session_state['last_estimate'] = estimacion_obj
                    st.session_state['last_tarifa'] = tarifa

    if 'last_estimate' in st.session_state:
        st.divider()
        st.subheader("Estimación Generada")

        est = st.session_state['last_estimate']
        tarifa_guardada = st.session_state['last_tarifa']
        
        tiempo_horas = est.get('tiempo_horas', 0.0)
        justificacion = est.get('justificacion', 'No se proporcionó justificación.')
        dificultad = est.get('dificultad', 'No especificada.')

        coste_mano_obra = tiempo_horas * tarifa_guardada

        col1, col2, col3 = st.columns(3)
        col1.metric("Tiempo Estimado", f"{tiempo_horas:.1f} horas")
        col2.metric("Coste Mano de Obra", f"{coste_mano_obra:.2f} €")
        col3.metric("Dificultad", dificultad)

        with st.expander("Ver justificación de la IA"):
            st.info(justificacion)

def history_tab():
    st.header("Historial de Consultas")
    try:
        logs = supabase.table("diagnostics_log").select("*").order("created_at", desc=True).limit(50).execute()
        if logs.data:
            df = pd.DataFrame(logs.data)
            df_display = df[['created_at', 'tecnico', 'modelo_maquina', 'descripcion_averia']]
            df_display['created_at'] = pd.to_datetime(df_display['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("No hay registros de consultas todavía.")
    except Exception as e:
        st.error(f"Error al cargar el historial: {e}")

def knowledge_base_tab():
    st.header("Base de Conocimiento")
    st.info("Aquí puedes ver todos los documentos que han sido procesados y cargados en la memoria del asistente.")

    try:
        response = supabase.table("documents").select("metadata", count="exact").execute()
        if response.data:
            # Extraer el nombre del fuente de cada metadata
            sources = [item['metadata']['source'] for item in response.data if 'metadata' in item and 'source' in item['metadata']]
            if sources:
                # Contar la frecuencia de cada fuente (que corresponde al número de chunks)
                df_counts = pd.Series(sources).value_counts().reset_index()
                df_counts.columns = ['Documento (Fuente)', 'Nº de Fragmentos']

                st.dataframe(df_counts, use_container_width=True, hide_index=True)
            else:
                st.warning("No se encontraron fuentes en los metadatos de los documentos.")
        else:
            st.info("La base de conocimiento está vacía. Carga algunos documentos PDF desde la barra lateral.")
    except Exception as e:
        st.error(f"Error al consultar la base de conocimiento: {e}")

def admin_sidebar():
    with st.sidebar:
        st.header("Administración")
        
        try:
            count_response = supabase.table("documents").select("id", count="exact").execute()
            doc_count = count_response.count
        except Exception as e:
            doc_count = f"Error: {e}"
        st.metric("Chunks en BBDD", doc_count)

        with st.expander("Cargar Manuales (PDF)"):
            pdf_files = st.file_uploader(
                "Sube uno o más archivos PDF",
                type=['pdf'],
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            if st.button("Procesar y Guardar PDFs"):
                if pdf_files:
                    with st.spinner("Procesando PDFs... Esto puede tardar varios minutos."):
                        ingest_pdf_files(pdf_files)
                else:
                    st.warning("Por favor, selecciona al menos un archivo PDF.")
        
# --- Aplicación Principal ---

def main():
    st.title("🔧 Asistente Técnico Satgarden")

    admin_sidebar()

    tab_titles = ["Consulta Técnica", "Calculadora", "Historial", "Base de Conocimiento"]
    tabs = st.tabs(tab_titles)

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

