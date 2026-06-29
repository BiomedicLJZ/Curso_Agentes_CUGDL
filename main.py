import os
import json
import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
from tavily import TavilyClient
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()
console = Console()

# ── Paths ──────────────────────────────────────────────────────────────────────
Raiz = Path(r"D:\Documentos\CUGDL\Curso_Agentes\Agente01")
MEMORIA_NARRATIVA = Raiz / "memoria.md"
CONCEPTOS_FILE = Raiz / "memorias" / "conceptos.json"
HISTORIAL_FILE = Raiz / "memorias" / "historial.json"
CONCEPTOS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Persistence helpers ────────────────────────────────────────────────────────
def _load_conceptos() -> dict:
    if CONCEPTOS_FILE.exists():
        with open(CONCEPTOS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_conceptos(data: dict) -> None:
    with open(CONCEPTOS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_historial() -> list:
    if not HISTORIAL_FILE.exists():
        return []
    with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for item in raw:
        if item["type"] == "human":
            out.append(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            out.append(AIMessage(content=item["content"]))
    return out


def _save_historial(history: list) -> None:
    data = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            data.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            data.append({"type": "ai", "content": msg.content})
    with open(HISTORIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── LLMs ──────────────────────────────────────────────────────────────────────
llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.6,
    top_p=0.95,
    max_tokens=16384,
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
)

llm2 = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.6,
    top_p=0.95,
    max_tokens=65536,
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
)


# ── Tools: web ────────────────────────────────────────────────────────────────
@tool
def search(query: str) -> dict:
    """Busca en internet usando Tavily. Devuelve resultados de la busqueda.
    :param query: consulta a buscar
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return client.search(query, max_results=20)


@tool
def crawl(query: str, url: str) -> dict:
    """Lee el contenido interno de una pagina web.
    :param query: que buscar en la pagina
    :param url: URL de la pagina
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return client.crawl(url, instructions=query)


@tool
def hora_actual(zona: str = "America/Mexico_City") -> str:
    """Devuelve la fecha y hora actual en la zona horaria indicada.
    :param zona: zona horaria IANA, p.ej. 'America/Mexico_City'
    """
    ahora = datetime.datetime.now(ZoneInfo(zona))
    return ahora.strftime("%Y-%m-%d %H:%M:%S %Z")


# ── Tools: memoria permanente ─────────────────────────────────────────────────
@tool
def leer_memoria() -> str:
    """Lee toda la memoria narrativa del agente desde memoria.md.
    Usala al inicio de cada conversacion para recordar informacion sobre el usuario."""
    if not MEMORIA_NARRATIVA.exists():
        return "Memoria narrativa vacia."
    return MEMORIA_NARRATIVA.read_text(encoding="utf-8")


@tool
def escribir_en_memoria(entrada: str) -> str:
    """Agrega informacion importante a la memoria narrativa (memoria.md).
    Usa para guardar hechos del usuario, conclusiones, o hallazgos relevantes.
    :param entrada: texto a agregar a la memoria
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(MEMORIA_NARRATIVA, "a", encoding="utf-8") as f:
        f.write(f"\n\n## [{ts}]\n{entrada}")
    return f"Memoria narrativa actualizada [{ts}]"


@tool
def guardar_concepto(clave: str, valor: str, categoria: str = "general") -> str:
    """Guarda un concepto estructurado clave-valor en memorias/conceptos.json.
    Usala para datos especificos: nombre del usuario, preferencias, proyectos activos.
    :param clave: identificador del concepto
    :param valor: contenido del concepto
    :param categoria: categoria para organizar
    """
    datos = _load_conceptos()
    datos.setdefault(categoria, {})[clave] = {
        "valor": valor,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    _save_conceptos(datos)
    return f"Concepto '{clave}' guardado en categoria '{categoria}'"


@tool
def leer_conceptos(categoria: str = "") -> str:
    """Lee conceptos almacenados en memorias/conceptos.json.
    :param categoria: filtrar por categoria especifica (vacio = todos los conceptos)
    """
    datos = _load_conceptos()
    if not datos:
        return "Sin conceptos almacenados."
    if categoria and categoria in datos:
        return json.dumps(datos[categoria], ensure_ascii=False, indent=2)
    return json.dumps(datos, ensure_ascii=False, indent=2)


@tool
def buscar_en_memoria(termino: str) -> str:
    """Busca un termino en toda la memoria del agente (narrativa + conceptos).
    :param termino: palabra o frase a buscar
    """
    resultados = []
    if MEMORIA_NARRATIVA.exists():
        lineas = [
            l for l in MEMORIA_NARRATIVA.read_text(encoding="utf-8").split("\n")
            if termino.lower() in l.lower()
        ]
        if lineas:
            resultados.append("Memoria narrativa:\n" + "\n".join(lineas))
    for cat, conceptos in _load_conceptos().items():
        for k, v in conceptos.items():
            if termino.lower() in k.lower() or termino.lower() in str(v.get("valor", "")).lower():
                resultados.append(f"[{cat}/{k}]: {v['valor']}")
    return "\n\n".join(resultados) if resultados else f"Sin resultados para '{termino}'"


@tool
def eliminar_concepto(categoria: str, clave: str) -> str:
    """Elimina un concepto de memorias/conceptos.json.
    :param categoria: categoria del concepto
    :param clave: clave del concepto a eliminar
    """
    datos = _load_conceptos()
    if categoria in datos and clave in datos[categoria]:
        del datos[categoria][clave]
        if not datos[categoria]:
            del datos[categoria]
        _save_conceptos(datos)
        return f"Concepto '{clave}' eliminado de '{categoria}'"
    return f"Concepto '{clave}' no encontrado en '{categoria}'"


# ── Tools: sistema de archivos local Windows ──────────────────────────────────
@tool
def listar_archivos(directorio: str = "") -> str:
    """Lista archivos y directorios en una ruta del sistema local Windows.
    :param directorio: ruta completa (vacio = directorio del proyecto)
    """
    path = Path(directorio) if directorio else Raiz
    if not path.exists():
        return f"Directorio no encontrado: {path}"
    items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = [f"{'[DIR] ' if p.is_dir() else '[FILE]'} {p.name}" for p in items]
    return f"Contenido de {path}:\n" + "\n".join(lines)


@tool
def leer_archivo_local(ruta: str) -> str:
    """Lee el contenido de cualquier archivo de texto en el sistema local Windows.
    :param ruta: ruta completa al archivo
    """
    path = Path(ruta)
    if not path.exists():
        return f"Archivo no encontrado: {ruta}"
    if path.is_dir():
        return f"'{ruta}' es un directorio, usa listar_archivos"
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Archivo binario o no-UTF8: {ruta}"
    except Exception as e:
        return f"Error leyendo {ruta}: {e}"


@tool
def escribir_archivo_local(ruta: str, contenido: str) -> str:
    """Escribe o crea un archivo de texto en el sistema local Windows.
    :param ruta: ruta completa al archivo (se crea si no existe)
    :param contenido: texto a escribir
    """
    path = Path(ruta)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contenido, encoding="utf-8")
    return f"Archivo guardado: {ruta} ({len(contenido):,} caracteres)"


# ── Sub-agent: Investigador ───────────────────────────────────────────────────
def _make_investigador() -> AgentExecutor:
    sub_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres investigador meticuloso. Resume hallazgos en vinetas breves y precisas."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    sub_agent = create_tool_calling_agent(llm2, [search, crawl], sub_prompt)
    return AgentExecutor(agent=sub_agent, tools=[search, crawl], verbose=False, max_iterations=8)


@tool
def investigar(pregunta: str) -> str:
    """Delega investigacion web profunda a un sub-agente especializado.
    Usalo para temas que requieren multiples busquedas o analisis de paginas.
    :param pregunta: tema o pregunta a investigar
    """
    executor = _make_investigador()
    result = executor.invoke({"input": pregunta, "chat_history": []})
    return result.get("output", "Sin resultado del investigador")


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Eres asistente IA para resolver dudas de usuarios.
Responde como cavernicola: evade articulos, conectores, conserva solo la esencia.

MEMORIA PERMANENTE - reglas obligatorias:
1. Al INICIO de cada conversacion llama leer_memoria() y leer_conceptos() para recordar al usuario
2. Cuando usuario comparta datos importantes (nombre, profesion, preferencias, proyectos)
   -> usa guardar_concepto() para datos estructurados
   -> usa escribir_en_memoria() para contexto narrativo
3. Al FINAL de conversaciones largas llama escribir_en_memoria() con resumen de conclusiones
4. Ante preguntas sobre temas discutidos antes usa buscar_en_memoria() antes de responder

HERRAMIENTAS DISPONIBLES:
- search / crawl: buscar y leer internet (Tavily)
- investigar: sub-agente investigador para temas complejos
- hora_actual: fecha/hora actual
- leer_memoria / escribir_en_memoria: memoria narrativa en disco (memoria.md)
- guardar_concepto / leer_conceptos / buscar_en_memoria / eliminar_concepto: memoria estructurada (memorias/conceptos.json)
- listar_archivos / leer_archivo_local / escribir_archivo_local: sistema archivos local Windows
"""

# ── Prompt template ────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ── All tools ─────────────────────────────────────────────────────────────────
TODAS_LAS_HERRAMIENTAS = [
    hora_actual,
    search,
    crawl,
    investigar,
    leer_memoria,
    escribir_en_memoria,
    guardar_concepto,
    leer_conceptos,
    buscar_en_memoria,
    eliminar_concepto,
    listar_archivos,
    leer_archivo_local,
    escribir_archivo_local,
]

# ── Agent setup ────────────────────────────────────────────────────────────────
agent_runnable = create_tool_calling_agent(llm, TODAS_LAS_HERRAMIENTAS, prompt)
agent = AgentExecutor(
    agent=agent_runnable,
    tools=TODAS_LAS_HERRAMIENTAS,
    verbose=True,
    max_iterations=20,
    handle_parsing_errors=True,
)

# ── Session history (persists across sessions via HISTORIAL_FILE) ──────────────
chat_history = _load_historial()


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global chat_history
    console.print(Panel(
        "[bold white]Agente LangChain — Memoria Permanente en Windows[/bold white]\n"
        "[dim]Comandos: 'salir' para terminar[/dim]",
        style="bold green",
        expand=False,
    ))

    while True:
        try:
            pregunta = input("\n✓ Escribe tu mensaje: ").strip()
            if not pregunta:
                continue
            if pregunta.lower() in ("salir", "exit", "quit"):
                console.print("[yellow]Guardando historial y saliendo...[/yellow]")
                _save_historial(chat_history[-20:])
                break

            response = agent.invoke({
                "input": pregunta,
                "chat_history": chat_history,
            })

            output = response.get("output", "")

            chat_history.append(HumanMessage(content=pregunta))
            chat_history.append(AIMessage(content=output))

            if len(chat_history) > 40:
                chat_history = chat_history[-40:]

            _save_historial(chat_history[-20:])

            console.print()
            console.rule("[dim]RESPUESTA[/dim]")
            console.print(Markdown(output))
            console.rule()

            steps = response.get("intermediate_steps", [])
            if steps:
                console.print(
                    f"[dim]Herramientas usadas: {', '.join(a.tool for a, _ in steps)}[/dim]"
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrumpido — guardando historial...[/yellow]")
            _save_historial(chat_history[-20:])
            break
        except Exception as exc:
            console.print(f"[red bold]Error:[/red bold] {exc}")


if __name__ == "__main__":
    main()
