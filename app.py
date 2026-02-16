import streamlit as st
import os
from typing import List

# Third-party imports
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool, BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# --- CONSTANTES Y CONFIGURACIÓN ---
DOWNLOAD_DIR = "downloads"
# ACTUALIZADO: Usamos el nuevo modelo soportado por Groq
MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Agente Investigador Seguro", page_icon="🕵️‍♂️")

# Asegurar que el directorio de descargas existe
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
        # SEGURIDAD: Sanitizar el nombre del archivo para evitar Path Traversal
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
    """Inicializa y retorna la lista de herramientas disponibles."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return [wiki_tool, save_to_file]

# --- LÓGICA DEL AGENTE (CACHEADA) ---

@st.cache_resource(show_spinner="Iniciando Agente IA...")
def init_agent(api_key: str):
    """
    Inicializa el LLM y el Agente. Usamos cache_resource para evitar 
    recrear el objeto en cada rerun de Streamlit.
    """
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=MODEL_NAME,
        temperature=0
    )

    tools = get_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Eres un asistente de investigación riguroso. Tu objetivo es buscar información precisa en Wikipedia. "
         "Si el usuario pide guardar información, USA la herramienta 'save_to_file'. "
         "Nunca inventes nombres de archivos fuera de contexto."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- INTERFAZ DE USUARIO ---

def main():
    st.title("🕵️‍♂️ Agente: Wikipedia a Archivo Seguro")

    # Sidebar
    with st.sidebar:
        st.header("Configuración")
        groq_api_key = st.text_input("Groq API Key", type="password")
        st.markdown("[Consigue tu API Key aquí](https://console.groq.com/keys)")
        
        # Botón para limpiar historial
        if st.button("Limpiar Historial"):
            st.session_state.messages = []
            st.rerun()

    # Validación de API Key
    if not groq_api_key:
        st.info("👈 Por favor, introduce tu API Key de Groq para comenzar.")
        return

    # Inicialización del Agente (Solo se ejecuta una vez gracias al caché)
    try:
        agent_executor = init_agent(groq_api_key)
    except Exception as e:
        st.error(f"Error al inicializar el agente. Verifica tu API Key. Detalles: {e}")
        return

    # Historial de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del Usuario
    if prompt_input := st.chat_input("Ej: Busca sobre 'Alan Turing' y guarda un resumen"):
        
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            try:
                # Usamos un spinner para mejor UX
                with st.spinner("Investigando y procesando..."):
                    response = agent_executor.invoke({"input": prompt_input})
                    output_text = response["output"]
                
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                # Verificación opcional: Mostrar archivos generados recientemente
                if "guardado" in output_text.lower():
                    if os.path.exists(DOWNLOAD_DIR):
                        files = os.listdir(DOWNLOAD_DIR)
                        if files:
                            st.success(f"Archivos disponibles en ./{DOWNLOAD_DIR}: {', '.join(files)}")

            except Exception as e:
                st.error(f"Ocurrió un error durante la ejecución: {e}")

if __name__ == "__main__":
    main()