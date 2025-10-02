"""
ASISTENTE DE DIAGNÓSTICO TÉCNICO - SATGARDEN MVP
Stack: OpenAI (LLM + Embeddings) + Supabase (pgvector) + Streamlit

Instalación:
pip install openai supabase streamlit pypdf2 pandas pillow python-dotenv
"""

import os
import streamlit as st
from openai import OpenAI
from supabase import create_client, Client
import PyPDF2
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

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
EMBEDDING_MODEL = "text-embedding-3-small"  # Más barato y rápido
EMBEDDING_DIMENSION = 1536

# ============================================
# FUNCIONES DE INGESTIÓN DE DATOS
# ============================================

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un PDF"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Divide texto en chunks con overlap"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Buscar el último punto para no cortar frases
        if end < text_length:
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:  # Al menos 70% del chunk
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def generate_embedding(text):
    """Genera embedding con OpenAI"""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def store_document(content, metadata):
    """Guarda documento en Supabase con su embedding"""
    embedding = generate_embedding(content)
    
    data = {
        "content": content,
        "metadata": metadata,
        "embedding": embedding
    }
    
    result = supabase.table("documents").insert(data).execute()
    return result

def ingest_pdf(pdf_path, doc_type="manual"):
    """Procesa PDF completo y lo guarda en Supabase"""
    filename = os.path.basename(pdf_path)
    st.info(f"Procesando: {filename}")
    
    # Extraer texto
    text = extract_text_from_pdf(pdf_path)
    
    # Dividir en chunks
    chunks = chunk_text(text)
    
    # Guardar cada chunk
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": filename,
            "type": doc_type,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        store_document(chunk, metadata)
        progress_bar.progress((i + 1) / len(chunks))
    
    st.success(f"✅ {filename}: {len(chunks)} chunks guardados")

def ingest_csv(csv_path):
    """Procesa CSV de histórico de reparaciones"""
    df = pd.read_csv(csv_path)
    st.info(f"Procesando {len(df)} registros de reparaciones")
    
    progress_bar = st.progress(0)
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
        
        store_document(content, metadata)
        progress_bar.progress((i + 1) / len(df))
    
    st.success(f"✅ {len(df)} casos de reparación guardados")

# ============================================
# FUNCIONES DE BÚSQUEDA Y DIAGNÓSTICO
# ============================================

def search_similar_documents(query, top_k=5):
    """Busca documentos similares en Supabase"""
    # Generar embedding de la query
    query_embedding = generate_embedding(query)
    
    # Llamar a la función de Supabase
    result = supabase.rpc(
        'match_documents',
        {
            'query_embedding': query_embedding,
            'match_count': top_k
        }
    ).execute()
    
    return result.data

def generate_diagnostic(query, context_docs):
    """Genera diagnóstico usando GPT-4 con contexto"""
    
    # Construir contexto desde documentos recuperados
    context = "\n\n---\n\n".join([
        f"Fuente: {doc['metadata'].get('source', 'Desconocida')}\n{doc['content']}"
        for doc in context_docs
    ])
    
    # Prompt especializado
    system_prompt = """Eres un técnico experto de Satgarden especializado en diagnóstico de maquinaria agrícola.

Tu trabajo es analizar averías y proporcionar diagnósticos precisos basándote en:
- Manuales técnicos de los fabricantes
- Histórico de reparaciones similares
- Tu conocimiento técnico

FORMATO DE RESPUESTA (usa siempre esta estructura):

## 🔍 Diagnósticos Probables
[Lista ordenada por probabilidad, cada uno con explicación breve]

## 🔧 Piezas Necesarias
[Lista con códigos de pieza si están disponibles]

## 📋 Procedimiento de Reparación
[Pasos numerados, claros y concisos]

## ⏱️ Estimación
- Tiempo: [minutos/horas]
- Dificultad: [Baja/Media/Alta]
- Coste piezas: [estimación]

## ❓ Información Adicional Necesaria
[Si necesitas más datos del técnico para afinar el diagnóstico]

Sé específico, práctico y cita las fuentes cuando sea relevante."""

    # Llamada a OpenAI
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXTO TÉCNICO:\n{context}\n\nCONSULTA DEL TÉCNICO:\n{query}"}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    
    return response.choices[0].message.content

def log_diagnostic(tecnico, modelo, descripcion, diagnostico, fue_util=None):
    """Registra diagnóstico para análisis posterior"""
    data = {
        "tecnico": tecnico,
        "modelo_maquina": modelo,
        "descripcion_averia": descripcion,
        "diagnostico_ia": diagnostico,
        "fue_util": fue_util
    }
    supabase.table("diagnostics_log").insert(data).execute()

# ============================================
# INTERFAZ STREAMLIT
# ============================================

def main():
    st.set_page_config(
        page_title="Asistente Diagnóstico Satgarden",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Asistente de Diagnóstico Técnico")
    st.markdown("**Satgarden** | Sistema RAG con IA")
    
    # Sidebar para gestión de datos
    with st.sidebar:
        st.header("⚙️ Administración")
        
        with st.expander("📤 Cargar Documentos"):
            st.subheader("Manuales PDF")
            uploaded_pdfs = st.file_uploader(
                "Sube manuales técnicos",
                type=['pdf'],
                accept_multiple_files=True
            )
            if uploaded_pdfs and st.button("Procesar PDFs"):
                for pdf_file in uploaded_pdfs:
                    # Guardar temporalmente
                    temp_path = f"temp_{pdf_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    
                    ingest_pdf(temp_path, doc_type="manual")
                    os.remove(temp_path)
            
            st.divider()
            
            st.subheader("Histórico CSV")
            uploaded_csv = st.file_uploader(
                "Sube histórico de reparaciones",
                type=['csv']
            )
            if uploaded_csv and st.button("Procesar CSV"):
                temp_path = f"temp_{uploaded_csv.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_csv.getbuffer())
                
                ingest_csv(temp_path)
                os.remove(temp_path)
        
        st.divider()
        
        # Estadísticas
        st.header("📊 Estadísticas")
        try:
            doc_count = supabase.table("documents").select("id", count="exact").execute()
            st.metric("Documentos en base", doc_count.count)
        except:
            st.metric("Documentos en base", "N/A")
    
    # Interfaz principal
    tabs = st.tabs(["🔍 Diagnóstico", "📚 Búsqueda", "📝 Historial"])
    
    # TAB 1: DIAGNÓSTICO
    with tabs[0]:
        st.header("Nuevo Diagnóstico")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tecnico = st.text_input("Nombre del técnico", value="", key="tecnico")
            
            modelo = st.selectbox(
                "Modelo de máquina",
                ["INFACO F3020", "SIS-350", "SIS-500", "AMB Rousset CR100", 
                 "Robot cortacésped", "Cosechadora", "Otro"],
                key="modelo"
            )
            
            if modelo == "Otro":
                modelo_custom = st.text_input("Especifica modelo:")
                if modelo_custom:
                    modelo = modelo_custom
        
        with col2:
            tipo_averia = st.selectbox(
                "Categoría",
                ["No arranca", "Ruido anormal", "Pérdida de potencia",
                 "Fuga", "Vibración", "Eléctrico", "Otro"]
            )
        
        descripcion = st.text_area(
            "Describe la avería en detalle",
            placeholder="Ejemplo: Motor no arranca. Al girar la llave hace clic repetitivo. Las luces del panel parpadean débilmente. Batería medida a 11.8V...",
            height=120
        )
        
        # Foto opcional
        uploaded_image = st.file_uploader(
            "📸 Foto de la avería (opcional)",
            type=['png', 'jpg', 'jpeg']
        )
        if uploaded_image:
            st.image(uploaded_image, width=300)
        
        if st.button("🔍 Generar Diagnóstico", type="primary", use_container_width=True):
            if not descripcion:
                st.error("Por favor, describe la avería")
            else:
                # Construir query completa
                query = f"""
                Modelo: {modelo}
                Categoría: {tipo_averia}
                Descripción: {descripcion}
                """
                
                with st.spinner("Analizando avería y consultando base de conocimiento..."):
                    # Buscar documentos relevantes
                    similar_docs = search_similar_documents(query, top_k=5)
                    
                    if not similar_docs:
                        st.warning("⚠️ No se encontraron documentos relevantes. Respuesta basada en conocimiento general.")
                        similar_docs = []
                    
                    # Generar diagnóstico
                    diagnostico = generate_diagnostic(query, similar_docs)
                    
                    # Registrar en log
                    log_diagnostic(tecnico or "Anónimo", modelo, descripcion, diagnostico)
                
                # Mostrar resultado
                st.success("✅ Diagnóstico completado")
                st.markdown(diagnostico)
                
                # Feedback
                st.divider()
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    if st.button("👍 Diagnóstico útil"):
                        st.success("¡Gracias por el feedback!")
                with col_fb2:
                    if st.button("👎 No fue útil"):
                        feedback = st.text_input("¿Qué faltó?")
                        st.info("Feedback registrado para mejorar")
                
                # Mostrar fuentes consultadas
                if similar_docs:
                    with st.expander("📚 Fuentes consultadas"):
                        for i, doc in enumerate(similar_docs):
                            st.markdown(f"**Fuente {i+1}:** {doc['metadata'].get('source', 'Desconocida')} (Relevancia: {doc['similarity']:.2%})")
                            st.text(doc['content'][:300] + "...")
                            st.divider()
    
    # TAB 2: BÚSQUEDA
    with tabs[1]:
        st.header("Búsqueda en Base de Conocimiento")
        
        search_query = st.text_input(
            "¿Qué información buscas?",
            placeholder="Ej: procedimiento cambio aceite SIS-350"
        )
        
        if st.button("Buscar"):
            if search_query:
                results = search_similar_documents(search_query, top_k=10)
                
                st.write(f"**{len(results)} resultados encontrados**")
                
                for i, doc in enumerate(results):
                    with st.expander(f"Resultado {i+1} - {doc['metadata'].get('source', 'Desconocida')} (Relevancia: {doc['similarity']:.2%})"):
                        st.markdown(doc['content'])
    
    # TAB 3: HISTORIAL
    with tabs[2]:
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
