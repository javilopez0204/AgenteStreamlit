import streamlit as st
import os
import asyncio
from typing import List

# --- IMPORTACIONES DE LANGCHAIN Y GOOGLE ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.tools import tool, BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# --- PARCHE PARA ASYNCIO EN STREAMLIT ---
# Soluciona el error "There is no current event loop"
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ----------------------------------------

# --- CONSTANTES ---
DOWNLOAD_DIR = "downloads"
MODEL_NAME = "gemini-2.0-flash"  # Modelo rápido y actual

st.set_page_config(page_title="Agente Investigador Gemini", page_icon="🤖")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- HERRAMIENTAS ---

@tool
def save_to_file(content: str, filename: str) -> str:
    """
    Guarda texto en un archivo local.
    Args:
        content: Texto a guardar.
        filename: Nombre del archivo (ej: 'resumen.txt').
    """
    try:
        safe_filename = os.path.basename(filename) or "resultado.txt"
        file_path = os.path.join(DOWNLOAD_DIR, safe_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Éxito: Archivo guardado en {file_path}"
    except Exception as e:
        return f"Error al guardar: {str(e)}"

def get_tools() -> List[BaseTool]:
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return [wiki_tool, save_to_file]

# --- INICIALIZACIÓN DEL AGENTE ---

@st.cache_resource(show_spinner="Conectando con Gemini...")
def init_agent(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=8192
    )
    
    tools = get_tools()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Eres un investigador experto. Tu misión es: "
         "1. Buscar información veraz en Wikipedia. "
         "2. Si el usuario lo pide, guardar un resumen usando 'save_to_file'. "
         "3. Ser conciso y profesional."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- INTERFAZ PRINCIPAL ---

def main():
    st.title("🤖 Agente: Wikipedia + Gemini 2.0")

    # Sidebar
    with st.sidebar:
        st.header("Configuración")
        google_api_key = st.text_input("Google AI Key", type="password")
        st.markdown("[Conseguir Key Gratis](https://aistudio.google.com/app/apikey)")
        if st.button("Limpiar Chat"):
            st.session_state.messages = []
            st.rerun()

    if not google_api_key:
        st.warning("👈 Por favor, introduce tu API Key para empezar.")
        return

    try:
        agent_executor = init_agent(google_api_key)
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return

    # Historial
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Usuario
    if prompt_input := st.chat_input("Ej: Investiga sobre 'Nikola Tesla' y guarda un resumen"):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            try:
                # Callback para ver el proceso de pensamiento
                st_callback = StreamlitCallbackHandler(st.container())
                
                response = agent_executor.invoke(
                    {"input": prompt_input},
                    {"callbacks": [st_callback]}
                )
                
                output_text = response["output"]
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                # --- NUEVA LÓGICA DE DESCARGA ---
                # Detectamos si se ha guardado algo y ofrecemos el archivo más reciente
                if "guardad" in output_text.lower() or "archivo" in output_text.lower():
                    if os.path.exists(DOWNLOAD_DIR):
                        files = os.listdir(DOWNLOAD_DIR)
                        if files:
                            # Encontrar el archivo más nuevo en la carpeta
                            latest_file_path = max(
                                [os.path.join(DOWNLOAD_DIR, f) for f in files], 
                                key=os.path.getctime
                            )
                            latest_filename = os.path.basename(latest_file_path)

                            # Leer el archivo para el botón
                            with open(latest_file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()

                            st.success(f"✅ Archivo generado: {latest_filename}")
                            
                            # Botón de descarga
                            st.download_button(
                                label=f"⬇️ Descargar {latest_filename}",
                                data=file_content,
                                file_name=latest_filename,
                                mime="text/plain"
                            )
                # --------------------------------

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()