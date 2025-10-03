"""
ASISTENTE TÉCNICO SATGARDEN MVP
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
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += re.sub(r'\s+', ' ', page_text.replace('\x00', '')) + "\n\n"
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return text.strip()

def chunk_text(text, chunk_size=1800):
    if not text or len(text) < 100:
        return []
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph if len(paragraph) <= chunk_size else paragraph[:chunk_size]
        else:
            current_chunk = (current_chunk + "\n\n" + paragraph) if current_chunk else paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) > 100]

def generate_embedding(text):
    try:
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text[:8000])
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def store_document(content, metadata):
    try:
        embedding = generate_embedding(content)
        if not embedding:
            return None
        result = supabase.table("documents").insert({"content": content, "metadata": metadata, "embedding": embedding}).execute()
        return result
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def ingest_pdf(pdf_path, doc_type="manual"):
    filename = os.path.basename(pdf_path)
    st.info(f"Procesando: {filename}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        st.error("No se pudo extraer texto")
        return
    chunks = chunk_text(text)
    if not chunks:
        st.error("No se crearon chunks")
        return
    progress_bar = st.progress(0)
    success = 0
    for i, chunk in enumerate(chunks):
        if store_document(chunk, {"source": filename, "type": doc_type, "chunk_index": i}):
            success += 1
        progress_bar.progress((i + 1) / len(chunks))
    st.success(f"{success} chunks guardados")

def ingest_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        progress_bar = st.progress(0)
        success = 0
        for i, row in df.iterrows():
            content = f"Modelo: {row.get('modelo', 'N/A')}\nAvería: {row.get('averia', 'N/A')}\nDiagnóstico: {row.get('diagnostico', 'N/A')}"
            if store_document(content, {"source": "historico", "type": "repair_case"}):
                success += 1
            progress_bar.progress((i + 1) / len(df))
        st.success(f"{success} casos guardados")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def search_similar_documents(query, top_k=5):
    try:
        emb = generate_embedding(query)
        if not emb:
            return []
        result = supabase.rpc('match_documents', {'query_embedding': emb, 'match_count': top_k}).execute()
        return result.data if result.data else []
    except:
        return []

def generate_technical_response(query, context_docs, tipo):
    context = "\n\n".join([f"{doc['metadata'].get('source', 'X')}: {doc['content']}" for doc in context_docs]) if context_docs else "Sin docs"
    prompts = {
        "Mantenimiento": "Mantenimiento estructurado",
        "Recambios": "Recambios con códigos",
        "Despiece": "Componentes y ubicaciones",
        "Avería": "Diagnóstico por probabilidad"
    }
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": f"Técnico Satgarden. {prompts.get(tipo, 'Procedimientos')}"}, {"role": "user", "content": f"CONTEXTO:\n{context}\n\nCONSULTA:\n{query}"}],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def extract_parts_from_image(image_file, modelo):
    try:
        img_bytes = image_file.read()
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        img_type = image_file.type.split('/')[-1]
        prompt = f'Analiza despiece de {modelo}. Extrae piezas en JSON: {{"modelo_maquina": "{modelo}", "piezas": [{{"numero_posicion": "1", "codigo": "X", "nombre": "Y", "cantidad": "1"}}]}}'
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/{img_type};base64,{b64}", "detail": "high"}}]}],
            max_tokens=2000
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except:
        st.error("Error extrayendo piezas")
        return None

def save_parts_to_catalog(data, img_name):
    try:
        count = 0
        for p in data.get('piezas', []):
            result = supabase.table("piezas_catalogo").insert({
                "modelo_maquina": data.get('modelo_maquina', 'X'),
                "numero_pieza": str(p.get('numero_posicion', '')),
                "codigo_referencia": p.get('codigo', 'X'),
                "nombre_pieza": p.get('nombre', ''),
                "cantidad": str(p.get('cantidad', '1')),
                "imagen_source": img_name
            }).execute()
            if result:
                count += 1
        return count
    except:
        return 0

def search_parts_in_catalog(modelo=None, pieza=None):
    try:
        query = supabase.table("piezas_catalogo").select("*")
        if modelo:
            query = query.ilike("modelo_maquina", f"%{modelo}%")
        if pieza:
            query = query.or_(f"nombre_pieza.ilike.%{pieza}%,codigo_referencia.ilike.%{pieza}%")
        result = query.limit(50).execute()
        return result.data if result.data else []
    except:
        return []

def search_parts_online(modelo, pieza):
    try:
        prompt = f"Información repuesto:\nMáquina: {modelo}\nPieza: {pieza}\n\nProporciona códigos, proveedores, precios, alternativas."
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": "Experto repuestos agrícolas"}, {"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_budget_estimate(trabajo, modelo, desc):
    try:
        prompt = f'Presupuesto {trabajo} en {modelo}: {desc}\n\nJSON: {{"tiempo_horas": 2.5, "tiempo_justificacion": "X", "piezas": [], "dificultad": "Media"}}'
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except:
        return None

def generate_pdf_report(data, respuesta, fuentes):
    if not REPORTLAB_AVAILABLE:
        return None
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("CONSULTA TÉCNICA", styles['Title']), Paragraph(f"Satgarden - {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']), Spacer(1, 0.5*cm), Paragraph(f"Técnico: {data.get('tecnico', 'X')}", styles['Normal']), Paragraph(f"Modelo: {data.get('modelo', 'X')}", styles['Normal']), Spacer(1, 0.5*cm), Paragraph("<b>Consulta:</b>", styles['Heading2']), Paragraph(data.get('consulta', ''), styles['Normal']), Spacer(1, 0.5*cm), Paragraph("<b>Respuesta:</b>", styles['Heading2'])]
        for line in respuesta.replace('#', '').replace('**', '').split('\n'):
            if line.strip():
                story.append(Paragraph(line, styles['Normal']))
        doc.build(story)
        buffer.seek(0)
        return buffer
    except:
        return None

def log_diagnostic(tecnico, modelo, desc, diag, util=None):
    try:
        supabase.table("diagnostics_log").insert({"tecnico": tecnico, "modelo_maquina": modelo, "descripcion_averia": desc, "diagnostico_ia": diag, "fue_util": util}).execute()
    except:
        pass

def main():
    st.set_page_config(page_title="Asistente Satgarden", page_icon="🔧", layout="wide")
    st.title("🔧 Asistente Técnico Satgarden")
    
    with st.sidebar:
        st.header("Administración")
        with st.expander("Cargar Documentos"):
            pdfs = st.file_uploader("PDFs", type=['pdf'], accept_multiple_files=True, key="pdf")
            if pdfs and st.button("Procesar PDFs"):
                for pdf in pdfs:
                    temp = f"temp_{pdf.name}"
                    with open(temp, "wb") as f:
                        f.write(pdf.getbuffer())
                    ingest_pdf(temp)
                    try:
                        os.remove(temp)
                    except:
                        pass
            st.divider()
            csv = st.file_uploader("CSV", type=['csv'], key="csv")
            if csv and st.button("Procesar CSV"):
                temp = f"temp_{csv.name}"
                with open(temp, "wb") as f:
                    f.write(csv.getbuffer())
                ingest_csv(temp)
                try:
                    os.remove(temp)
                except:
                    pass
        st.divider()
        try:
            count = supabase.table("documents").select("id", count="exact").execute()
            st.metric("Chunks", count.count if count.count else 0)
        except:
            st.metric("Chunks", "Error")
    
    tabs = st.tabs(["Consulta", "Recambios", "Calculadora", "Dashboard", "Historial"])
    
    with tabs[0]:
        st.header("Consulta Técnica")
        col1, col2 = st.columns([3, 1])
        with col1:
            modelo = st.text_input("Modelo", key="modelo")
        with col2:
            tipo = st.selectbox("Tipo", ["Mantenimiento", "Avería", "Recambios", "Despiece"])
        consulta = st.text_area("¿Qué necesitas?", height=120, key="consulta")
        tecnico = st.text_input("Técnico", key="tecnico")
        if st.button("Buscar", type="primary", use_container_width=True):
            if consulta:
                query = f"Modelo: {modelo or 'X'}\nTipo: {tipo}\n{consulta}"
                with st.spinner("Buscando..."):
                    docs = search_similar_documents(query)
                    resp = generate_technical_response(query, docs, tipo)
                    log_diagnostic(tecnico or "Anónimo", modelo or "X", consulta, resp)
                st.success("Listo")
                st.markdown(resp)
                pdf = generate_pdf_report({'tecnico': tecnico or "X", 'modelo': modelo or "X", 'tipo': tipo, 'consulta': consulta}, resp, docs)
                if pdf:
                    st.download_button("Descargar PDF", pdf, f"consulta_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf", use_container_width=True)
    
    with tabs[1]:
        st.header("Recambios")
        mode = st.radio("Modo", ["Buscar", "Catalogar"], horizontal=True)
        if mode == "Buscar":
            col1, col2 = st.columns(2)
            with col1:
                mod_bus = st.text_input("Modelo", key="mod_bus")
            with col2:
                pieza_bus = st.text_input("Pieza", key="pieza_bus")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Buscar en Catálogo", use_container_width=True):
                    res = search_parts_in_catalog(mod_bus, pieza_bus)
                    if res:
                        st.success(f"{len(res)} encontradas")
                        df = pd.DataFrame(res)[['modelo_maquina', 'codigo_referencia', 'nombre_pieza']]
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No encontradas")
            with col_b:
                if st.button("Buscar Online", use_container_width=True):
                    if mod_bus and pieza_bus:
                        with st.spinner("Buscando..."):
                            res = search_parts_online(mod_bus, pieza_bus)
                        st.markdown(res)
        else:
            modelo_desp = st.text_input("Modelo", key="modelo_desp")
            img = st.file_uploader("Imagen despiece", type=['png', 'jpg', 'jpeg'], key="img_desp")
            if img:
                st.image(img, use_column_width=True)
                if st.button("Extraer Piezas", type="primary", use_container_width=True):
                    if modelo_desp:
                        with st.spinner("Analizando (GPT-4V)..."):
                            img.seek(0)
                            piezas = extract_parts_from_image(img, modelo_desp)
                        if piezas:
                            st.success(f"{len(piezas.get('piezas', []))} piezas")
                            df = pd.DataFrame(piezas.get('piezas', []))
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            col_s, col_d = st.columns(2)
                            with col_s:
                                if st.button("Guardar", use_container_width=True):
                                    saved = save_parts_to_catalog(piezas, img.name)
                                    st.success(f"{saved} guardadas")
                            with col_d:
                                csv = df.to_csv(index=False)
                                st.download_button("Descargar CSV", csv, f"despiece_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
    
    with tabs[2]:
        st.header("Calculadora")
        col1, col2 = st.columns([2, 1])
        with col1:
            mod_pres = st.text_input("Modelo", key="mod_pres")
        with col2:
            tipo_trab = st.selectbox("Tipo", ["Mantenimiento", "Reparación"], key="tipo_trab")
        desc_trab = st.text_area("Descripción", height=100, key="desc_trab")
        with st.expander("Config"):
            tarifa = st.number_input("Tarifa/h (€)", 20, 100, 45, 5, key="tarifa")
        if st.button("Calcular", type="primary", use_container_width=True):
            if desc_trab:
                with st.spinner("Calculando..."):
                    est = generate_budget_estimate(tipo_trab, mod_pres, desc_trab)
                if est:
                    st.success("Listo")
                    mano = est['tiempo_horas'] * tarifa
                    piezas_cost = sum([p['precio_estimado'] for p in est.get('piezas', [])])
                    total = mano + piezas_cost
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Tiempo", f"{est['tiempo_horas']} h")
                    with col_b:
                        st.metric("Mano obra", f"{mano:.2f}€")
                    with col_c:
                        st.metric("TOTAL", f"{total:.2f}€")
    
    with tabs[3]:
        st.header("Dashboard")
        try:
            logs = supabase.table("diagnostics_log").select("*").execute()
            if logs.data:
                df = pd.DataFrame(logs.data)
                df['created_at'] = pd.to_datetime(df['created_at'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total", len(df))
                with col2:
                    rec = df[df['created_at'] > pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)]
                    st.metric("Últimos 7 días", len(rec))
                with col3:
                    st.metric("Satisfacción", "N/A")
                st.divider()
                if 'modelo_maquina' in df.columns:
                    st.subheader("Máquinas consultadas")
                    st.bar_chart(df['modelo_maquina'].value_counts().head(5))
            else:
                st.info("Sin datos")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tabs[4]:
        st.header("Historial")
        try:
            logs = supabase.table("diagnostics_log").select("*").order("created_at", desc=True).limit(20).execute()
            if logs.data:
                df = pd.DataFrame(logs.data)
                st.dataframe(df[['created_at', 'tecnico', 'modelo_maquina']], use_container_width=True)
            else:
                st.info("Sin registros")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
