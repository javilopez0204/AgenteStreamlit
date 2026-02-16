import streamlit as st
import os
from typing import List

# --- NUEVAS IMPORTACIONES PARA GOOGLE ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool, BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# --- CONSTANTES Y CONFIGURACIÓN ---
DOWNLOAD_DIR = "downloads"
# Usamos Gemini 1.5 Flash que es excelente para tareas rápidas y gratuito
# Opciones: "gemini-1.5-flash" o "gemini-1.5-pro"
MODEL_NAME = "gemini-1.5-flash" 

st.set_page_config(page_title="Agente Investigador Gemini", page_icon="🤖")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- DEFINICIÓN DE HERRAMIENTAS ---

@tool
def save_to_file(content: str, filename: str) -> str:
    """
    Guarda texto en un archivo local dentro del directorio seguro 'downloads'.
    Args:
        content: El contenido de texto a guardar.
        filename: El nombre del archivo deseado (se sanitizará automáticamente).
    """
    try:
        safe_filename = os.path.basename(filename)
        if not safe_filename:
            safe_filename = "resultado_agente.txt"
            
        file_path = os.path.join(DOWNLOAD_DIR, safe_filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Éxito: Archivo guardado como '{safe_filename}' en el directorio de descargas."
    except Exception as e:
        return f"Error crítico al guardar el archivo: {str(e)}"

def get_tools() -> List[BaseTool]:
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return [wiki_tool, save_to_file]

# --- LÓGICA DEL AGENTE (CACHEADA) ---

@st.cache_resource(show_spinner="Iniciando Agente Gemini...")
def init_agent(api_key: str):
    """
    Inicializa el LLM de Google y el Agente.
    """
    # Configuramos el modelo de Google
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=8192
    )

    tools = get_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Eres un asistente de investigación experto y riguroso impulsado por Gemini. "
         "Tu objetivo es buscar información precisa en Wikipedia. "
         "Si encuentras la información relevante y el usuario pidió guardarla, DEBES usar la herramienta 'save_to_file'. "
         "Proporciona nombres de archivo descriptivos (ej: 'biografia_turing.txt')."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # create_tool_calling_agent funciona nativamente con Gemini
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- INTERFAZ DE USUARIO ---

def main():
    st.title("🤖 Agente: Wikipedia + Gemini")

    # Sidebar
    with st.sidebar:
        st.header("Configuración de Google")
        google_api_key = st.text_input("Google AI Studio Key", type="password")
        st.markdown("[Obtén tu API Key gratuita aquí](https://aistudio.google.com/app/apikey)")
        
        if st.button("Limpiar Historial"):
            st.session_state.messages = []
            st.rerun()

    # Validación de API Key
    if not google_api_key:
        st.info("👈 Introduce tu API Key de Google AI Studio para comenzar.")
        return

    try:
        agent_executor = init_agent(google_api_key)
    except Exception as e:
        st.error(f"Error al conectar con Google Gemini. Detalles: {e}")
        return

    # Historial de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del Usuario
    if prompt_input := st.chat_input("Ej: Investiga sobre la 'Computación Cuántica' y guarda un resumen"):
        
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Gemini está investigando..."):
                    response = agent_executor.invoke({"input": prompt_input})
                    output_text = response["output"]
                
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                # Verificación de archivos
                if "guardado" in output_text.lower() or "archivo" in output_text.lower():
                    if os.path.exists(DOWNLOAD_DIR):
                        files = os.listdir(DOWNLOAD_DIR)
                        if files:
                            st.success(f"📂 Archivos en carpeta segura: {', '.join(files)}")

            except Exception as e:
                st.error(f"Error durante la ejecución: {e}")

if __name__ == "__main__":
    main()