"""
ASISTENTE TÉCNICO SATGARDEN V2.0
Implementación de:
1. Sistema de Conocimiento Verificado
2. Dashboard de Inteligencia Técnica
3. Generador de Planes de Mantenimiento Preventivo
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

# --- Dependencias Opcionales ---
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
st.set_page_config(page_title="Asistente Satgarden V2", page_icon="🛠️", layout="wide")

# --- Conexiones (Cacheado para Rendimiento) ---
@st.cache_resource
def init_connections():
    if not all([os.getenv("OPENAI_API_KEY"), os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")]):
        st.error("Faltan variables de entorno. Revisa tu archivo .env.")
        st.stop()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return openai_client, supabase_client

openai_client, supabase = init_connections()
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Funciones de IA (OpenAI) ---
def generate_embedding(text):
    try:
        text = text.replace("\n", " ").strip()[:8191]
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Error al generar embedding: {e}")
        return None

# ... (El resto de funciones de IA como generate_technical_response, etc. se mantienen similares pero se llamarán desde las nuevas pestañas)

# --- Funciones de Base de Datos (Supabase) ---
def search_verified_knowledge(query_text, top_k=1, threshold=0.8):
    embedding = generate_embedding(query_text)
    if not embedding: return None
    try:
        result = supabase.rpc('match_verified_documents', {
            'query_embedding': embedding, 'match_count': top_k, 'match_threshold': threshold
        }).execute()
        return result.data[0] if result.data else None
    except Exception:
        return None # Falla silenciosamente si la función aún no existe

def search_document_knowledge(query_text, top_k=5):
    # ... (Función existente para buscar en la tabla 'documents')
    embedding = generate_embedding(query_text)
    if not embedding: return []
    try:
        result = supabase.rpc('match_documents', {
            'query_embedding': embedding, 'match_count': top_k
        }).execute()
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Error en la búsqueda de documentos: {e}")
        return []

def log_and_get_id(tecnico, modelo, tipo, desc, diag):
    try:
        response = supabase.table("diagnostics_log").insert({
            "tecnico": tecnico, "modelo_maquina": modelo, "tipo_consulta": tipo,
            "descripcion_averia": desc, "diagnostico_ia": diag
        }).select("id").execute()
        return response.data[0]['id'] if response.data else None
    except Exception as e:
        st.warning(f"No se pudo registrar el diagnóstico: {e}")
        return None

def update_feedback(log_id, feedback_value):
    try:
        supabase.table("diagnostics_log").update({"feedback": feedback_value}).eq("id", log_id).execute()
        st.toast("¡Gracias por tu feedback!")
    except Exception as e:
        st.error(f"Error al guardar feedback: {e}")

def save_verified_knowledge(query, response, verifier):
    embedding = generate_embedding(response)
    if embedding:
        try:
            supabase.table("verified_knowledge").insert({
                "original_query": query, "verified_response": response,
                "verified_by": verifier, "embedding": embedding
            }).execute()
            st.success("¡Respuesta verificada y guardada!")
        except Exception as e:
            st.error(f"Error al guardar conocimiento verificado: {e}")

# --- Pestañas de la UI ---

def consult_tab():
    st.header("Consulta Técnica")
    # ... (Lógica del formulario de consulta)
    if 'last_response' in st.session_state and st.button("Nueva Consulta"):
        # Limpiar estado para nueva consulta
        for key in ['last_response', 'last_query_data', 'context_docs', 'verified', 'log_id']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Formulario de consulta
    with st.form("consulta_form"):
        # ... (Inputs: modelo, tipo, consulta, tecnico)
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
                # ... (Llamada a generate_technical_response)
                respuesta_ia = f"Respuesta generada por IA para: {consulta}" # Placeholder
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

        # Feedback
        if not st.session_state.get('verified') and 'log_id' in st.session_state:
            log_id = st.session_state['log_id']
            cols = st.columns(10)
            cols[0].button("👍", on_click=update_feedback, args=(log_id, 1))
            cols[1].button("👎", on_click=update_feedback, args=(log_id, -1))

def maintenance_tab():
    st.header("Generador de Planes de Mantenimiento")
    # ... (Lógica para seleccionar máquina y horas, y generar plan)
    st.info("Función en desarrollo.")


def dashboard_tab():
    st.header("Dashboard de Inteligencia Técnica")
    try:
        logs = supabase.table("diagnostics_log").select("*").execute().data
        if not logs:
            st.info("No hay datos suficientes para generar el dashboard.")
            return

        df = pd.DataFrame(logs)
        st.subheader("Métricas Generales")
        # ... (Lógica para st.metric de satisfacción, etc.)
        feedback_counts = df['feedback'].value_counts()
        util = feedback_counts.get(1, 0)
        no_util = feedback_counts.get(-1, 0)
        satisfaction = (util / (util + no_util) * 100) if (util + no_util) > 0 else 0
        st.metric("Satisfacción de Respuestas IA", f"{satisfaction:.1f}%")

        st.subheader("Máquinas Más Consultadas")
        st.bar_chart(df['modelo_maquina'].value_counts())

        st.subheader("Tipos de Consulta Frecuentes")
        st.bar_chart(df['tipo_consulta'].value_counts())

    except Exception as e:
        st.error(f"Error al generar dashboard: {e}")


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
        verified_response = st.text_area("Edita la respuesta para guardarla como oficial:",
                                         value=log_to_verify['diagnostico_ia'], height=200)
        verifier = st.text_input("Tu nombre (verificador):")
        if st.button("Guardar Conocimiento Verificado"):
            if verifier:
                save_verified_knowledge(log_to_verify['descripcion_averia'], verified_response, verifier)
                del st.session_state['verify_item']
                st.rerun()
            else:
                st.warning("Por favor, introduce tu nombre como verificador.")

# --- Aplicación Principal ---
def main():
    st.title("🛠️ Asistente Técnico Satgarden V2.0")

    # ... (Barra lateral de administración se mantiene similar)

    tabs = st.tabs(["Consulta", "Mantenimiento Preventivo", "Dashboard", "Historial y Verificación", "Gestión de Conocimiento"])

    with tabs[0]:
        consult_tab()
    with tabs[1]:
        maintenance_tab()
    with tabs[2]:
        dashboard_tab()
    with tabs[3]:
        history_tab()
    with tabs[4]:
        # La pestaña de "knowledge_base_tab" se mantiene igual que en V1.3
        st.info("Pestaña de Gestión de Conocimiento en desarrollo.")

if __name__ == "__main__":
    main()
