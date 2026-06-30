# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo[mcp]",
#     "langchain>=1.0",
#     "langgraph>=1.0",
#     "langgraph-checkpoint-sqlite",
#     "langchain-nvidia-ai-endpoints",
#     "python-dotenv",         # carga NVIDIA_API_KEY/TAVILY_API_KEY desde un archivo .env
#     "tavily-python",         # cliente de las tools de busqueda web (search/research/extract)
#     "numpy",
#     "duckdb==1.5.4",
#     "sqlglot==30.12.0",
#     "polars[pyarrow]==1.42.0",
#     "mcp>=1",
#     "pydantic>=2",
#     "openai==2.44.0",
#     "ruff==0.15.20",
#     "altair==6.2.2",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
#     "nbformat==5.10.4",
#     "python-lsp-server==1.14.0",
#     "websockets==16.0",
#     "python-lsp-ruff==2.3.1",
#     "pytest==9.1.1",
#     "ty==0.0.55",
#     "basedpyright==1.39.9",
#     "pyrefly==1.1.1",
# ]
# ///
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EL CEREBRO EN EL FRASCO  ·  agente con memoria persistente local         ║
# ║                                                                          ║
# ║  Run:   uv run marimo edit cerebro_en_el_frasco.py                        ║
# ║  Need:  set NVIDIA_API_KEY  (export / $env:NVIDIA_API_KEY="nvapi-...")    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ── ARQUITECTURA DE MEMORIA ────────────────────────────────────────────────────
#
#   CAPA 1 · Corto plazo  (SqliteSaver)
#     Guarda el hilo exacto de la conversación actual. Se reinicia por sesión.
#     Permite que LangGraph recuerde el contexto dentro de un mismo chat.
#
#   CAPA 2 · Largo plazo  (SQLitePersistentStore — custom)
#     Almacena hechos duraderos del usuario entre TODOS los chats.
#     Búsqueda híbrida: semántica (coseno NumPy) + keyword (LIKE SQL) + recencia.
#     Sin extensiones C (sqlite-vec/vss) → funciona en Windows sin compilar nada.
#
#   CAPA 3 · Reflexión autónoma  (_reflexion_autonoma en run_agent)
#     Post-turn: usa el modelo de respaldo para extraer hechos nuevos del
#     intercambio y los persiste en silencio, sin que el usuario lo pida.
#     Complementa — no reemplaza — las llamadas a `recordar` del propio agente.
#
# ── HERRAMIENTAS (TOOLS) DEL AGENTE ────────────────────────────────────────────
#
#   Memoria (operan sobre el store de largo plazo):
#     · recordar(texto, categoria)  → graba un hecho duradero.   [ESCRITURA]
#     · evocar(consulta, modo)      → busca recuerdos.           [LECTURA]
#     · olvidar(consulta)           → borra el recuerdo que más coincide. [DESTRUCTIVA]
#   Web (vía API de Tavily, requieren TAVILY_API_KEY):
#     · search(query)        → búsqueda en Internet.
#     · research(query)      → investigación más profunda.
#     · extract_webpage(url) → extrae el contenido de una página.
#
# ── PIPELINE DE MIDDLEWARES (envuelven al agente; orden = afuera→adentro) ───────
#
#   inyectar_memoria (custom, @dynamic_prompt) · SIEMPRE primero:
#     inyecta los recuerdos relevantes en el system prompt antes de cada turno.
#   Resto (activables desde el menú interactivo `middleware_menu`):
#     summarization · context_editing · human_in_loop · model_call_limit ·
#     tool_call_limit · tool_retry · model_retry · model_fallback ·
#     todo_planning · tool_selector · pii
#   Nota: `human_in_loop` PAUSA el grafo esperando aprobación humana; viene
#   apagado por defecto porque la UI de chat de marimo no puede reanudarlo.
#
# ──────────────────────────────────────────────────────────────────────────────

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 El Cerebro en el Frasco
    Un agente `create_agent` (LangChain v1) movido por **NVIDIA NIM**, con
    memoria **persistente en tu disco** — corto plazo (hilo actual) vía
    `SqliteSaver`, y largo plazo (entre todos los chats) vía un *store*
    SQLite propio con búsqueda **semántica + por palabra clave + por recencia**.

    Edita los parámetros en la celda **CONFIG**, asegúrate de tener
    `NVIDIA_API_KEY`, y baja al chat.# 🧠 El Cerebro en el Frasco
    Un agente `create_agent` (LangChain v1) movido por **NVIDIA NIM**, con
    memoria **persistente en tu disco** — corto plazo (hilo actual) vía
    `SqliteSaver`, y largo plazo (entre todos los chats) vía un *store*
    SQLite propio con búsqueda **semántica + por palabra clave + por recencia**.

    Edita los parámetros en la celda **CONFIG**, asegúrate de tener
    `NVIDIA_API_KEY`, y baja al chat.
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    return Path, load_dotenv, os


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell
def _(Path, os):
    NVIDIA_MODEL = os.environ.get(
        "NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
    )
    FALLBACK_MODEL = os.environ.get(
        "NIM_FALLBACK", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    )
    EMB_MODEL = os.environ.get("NIM_EMBED", "nvidia/nv-embedqa-e5-v5")
    TEMPERATURE = float(os.environ.get("NIM_TEMPERATURE", "0.6"))

    # --- Identidad de la memoria a largo plazo (namespace) ---
    USER_ID = os.environ.get("CEREBRO_USER", "leonardo")

    # --- Dónde viven los .db. Anclado al HOME => nunca depende del cwd,
    #     que es la causa #1 silenciosa de "no me guardó nada" en Windows. ---
    MEMORY_DIR = Path(
        os.environ.get("CEREBRO_DIR", Path.home() / ".cerebro_memory")
    )
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ITEMS_DB_PATH = str((MEMORY_DIR / "memorias_largo_plazo.db").resolve())
    CP_DB_PATH = str((MEMORY_DIR / "hilos_corto_plazo.db").resolve())
    return (
        CP_DB_PATH,
        EMB_MODEL,
        FALLBACK_MODEL,
        ITEMS_DB_PATH,
        NVIDIA_MODEL,
        TEMPERATURE,
        USER_ID,
    )


@app.cell
def _():
    MIDDLEWARE_TOGGLES = {
        "summarization": True,  # resume historiales largos
        "human_in_loop": True,
        "context_editing": True,  # poda resultados viejos de tools
        "model_call_limit": True,  # tope de llamadas al modelo (seguridad)
        "tool_call_limit": True,  # tope de llamadas a tools
        "tool_retry": True,  # reintenta tools que fallan
        "model_retry": True,  # reintenta el modelo ante errores transitorios
        "model_fallback": True,  # cae al FALLBACK_MODEL si el principal falla
        "todo_planning": True,  # planificación tipo to-do para tareas complejas
        "tool_selector": False,  # útil sólo con MUCHAS tools; aquí son 3
        "pii": False,  # OFF: bloquearía la info personal del usuario
    }
    return


@app.cell
def _(os):
    NVIDIA_KEY_PRESENT = os.environ.get("NVIDIA_API_KEY").startswith("nvapi")
    return (NVIDIA_KEY_PRESENT,)


@app.cell
def _(mo):
    middleware_menu = mo.ui.dictionary(
        {
            "summarization": mo.ui.switch(
                value=True, label="**Summarization**: Resume historiales largos"
            ),
            "human_in_loop": mo.ui.switch(
                value=True,
                label="**Human in Loop**: Requiere aprobación para acciones de escritura/búsqueda",
            ),
            "context_editing": mo.ui.switch(
                value=True,
                label="**Context Editing**: Poda resultados viejos de tools",
            ),
            "model_call_limit": mo.ui.switch(
                value=True,
                label="**Model Call Limit**: Tope de llamadas al modelo (seguridad)",
            ),
            "tool_call_limit": mo.ui.switch(
                value=True,
                label="**Tool Call Limit**: Tope de llamadas a tools",
            ),
            "tool_retry": mo.ui.switch(
                value=True, label="**Tool Retry**: Reintenta tools que fallan"
            ),
            "model_retry": mo.ui.switch(
                value=True,
                label="**Model Retry**: Reintenta el modelo ante errores transitorios",
            ),
            "model_fallback": mo.ui.switch(
                value=True,
                label="**Model Fallback**: Cae al fallback_model si el principal falla",
            ),
            "todo_planning": mo.ui.switch(
                value=True,
                label="**To-do Planning**: Planificación tipo to-do para tareas complejas",
            ),
            "tool_selector": mo.ui.switch(
                value=False,
                label="**Tool Selector**: Selector dinámico (útil con MUCHAS tools)",
            ),
            "pii": mo.ui.switch(
                value=False,
                label="**PII**: Bloquea/enmascara info personal del usuario (email, url, ip, phone, ssn)",
            ),
        }
    )
    return (middleware_menu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ⚙️ Agent Middleware Settings
    *Configure the active middlewares for the current agent session.*
    """)
    return


@app.cell
def _(middleware_menu):
    middleware_menu
    return


@app.cell
def _():
    import sqlite3, json, uuid, datetime
    import numpy as np

    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        dynamic_prompt,
        SummarizationMiddleware,
        ContextEditingMiddleware,
        HumanInTheLoopMiddleware,
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
        ModelRetryMiddleware,
        ModelFallbackMiddleware,
        LLMToolSelectorMiddleware,
        TodoListMiddleware,
        PIIMiddleware,
    )
    from langchain.tools import tool
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.store.base import Item, PutOp

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

        NVIDIA_PKG = True
    except Exception:  # paquete no instalado
        NVIDIA_PKG = False
    return (
        ChatNVIDIA,
        ContextEditingMiddleware,
        HumanInTheLoopMiddleware,
        InMemoryStore,
        Item,
        LLMToolSelectorMiddleware,
        ModelCallLimitMiddleware,
        ModelFallbackMiddleware,
        ModelRetryMiddleware,
        NVIDIAEmbeddings,
        NVIDIA_PKG,
        PIIMiddleware,
        PutOp,
        SqliteSaver,
        SummarizationMiddleware,
        TodoListMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
        create_agent,
        datetime,
        dynamic_prompt,
        json,
        np,
        sqlite3,
        tool,
        uuid,
    )


@app.cell
def _(InMemoryStore, Item, PutOp, json, np, sqlite3):
    class SQLitePersistentStore(InMemoryStore):
        """InMemoryStore + persistencia SQLite. Sin extensiones. Windows-safe."""

        def __init__(self, db_path, *, index=None):
            super().__init__(index=index)
            self._db_path = str(db_path)
            # check_same_thread=False: LangGraph toca la conexión desde hilos
            # worker; SQLite lo prohíbe por defecto y lanza error.
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_items(
                    namespace TEXT, key TEXT, value TEXT,
                    created_at TEXT, updated_at TEXT,
                    PRIMARY KEY(namespace, key));
                CREATE TABLE IF NOT EXISTS store_vectors(
                    namespace TEXT, key TEXT, path TEXT, vector BLOB,
                    PRIMARY KEY(namespace, key, path));
                """
            )
            self._conn.commit()
            self._load()

        @staticmethod
        def _n2s(ns):
            return json.dumps(list(ns))

        @staticmethod
        def _s2n(s):
            return tuple(json.loads(s))

        def _load(self):
            for ns_s, key, value, ca, ua in self._conn.execute(
                "SELECT namespace,key,value,created_at,updated_at FROM store_items"
            ):
                ns = self._s2n(ns_s)
                self._data[ns][key] = Item(
                    value=json.loads(value),
                    key=key,
                    namespace=ns,
                    created_at=ca,
                    updated_at=ua,
                )
            for ns_s, key, path, blob in self._conn.execute(
                "SELECT namespace,key,path,vector FROM store_vectors"
            ):
                ns = self._s2n(ns_s)
                self._vectors[ns][key][path] = np.frombuffer(
                    blob, dtype=np.float32
                ).tolist()

        def _persist(self, ops):
            c = self._conn
            for op in ops:
                if not isinstance(op, PutOp):
                    continue
                ns_s = self._n2s(op.namespace)
                if op.value is None:  # borrado
                    c.execute(
                        "DELETE FROM store_items   WHERE namespace=? AND key=?",
                        (ns_s, op.key),
                    )
                    c.execute(
                        "DELETE FROM store_vectors WHERE namespace=? AND key=?",
                        (ns_s, op.key),
                    )
                    continue
                it = self._data[op.namespace][op.key]  # Item recién aplicado
                c.execute(
                    "INSERT OR REPLACE INTO store_items VALUES(?,?,?,?,?)",
                    (
                        ns_s,
                        op.key,
                        json.dumps(it.value),
                        str(it.created_at),
                        str(it.updated_at),
                    ),
                )
                c.execute(
                    "DELETE FROM store_vectors WHERE namespace=? AND key=?",
                    (ns_s, op.key),
                )
                for path, vec in (
                    self._vectors.get(op.namespace, {}).get(op.key, {}).items()
                ):
                    c.execute(
                        "INSERT OR REPLACE INTO store_vectors VALUES(?,?,?,?)",
                        (
                            ns_s,
                            op.key,
                            path,
                            np.asarray(vec, dtype=np.float32).tobytes(),
                        ),
                    )
            c.commit()

        def batch(self, ops):
            ops = list(ops)
            res = super().batch(ops)  # embeddings + búsqueda + apply
            self._persist(ops)  # write-through
            return res

        async def abatch(self, ops):
            ops = list(ops)
            res = await super().abatch(ops)
            self._persist(ops)
            return res

        # --- recall extra que NO necesita embeddings (SQL directo) ---
        def recent(self, namespace, limit=5):
            rows = self._conn.execute(
                "SELECT value FROM store_items WHERE namespace=? "
                "ORDER BY updated_at DESC LIMIT ?",
                (self._n2s(namespace), limit),
            ).fetchall()
            return [json.loads(v) for (v,) in rows]

        def keyword(self, namespace, term, limit=10):
            rows = self._conn.execute(
                "SELECT value FROM store_items WHERE namespace=? "
                "AND lower(value) LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (self._n2s(namespace), f"%{term.lower()}%", limit),
            ).fetchall()
            return [json.loads(v) for (v,) in rows]

    return (SQLitePersistentStore,)


@app.cell
def _(
    EMB_MODEL,
    ITEMS_DB_PATH,
    NVIDIAEmbeddings,
    NVIDIA_KEY_PRESENT,
    NVIDIA_PKG,
    SQLitePersistentStore,
    USER_ID,
):
    MEM_NS = ("memorias", USER_ID)
    embedder, EMB_DIMS, semantic_ok = None, None, False
    if NVIDIA_PKG and NVIDIA_KEY_PRESENT:
        try:
            embedder = NVIDIAEmbeddings(model=EMB_MODEL, truncate="END")
            EMB_DIMS = len(
                embedder.embed_query("sonda de dimensiones")
            )  # probe
            semantic_ok = True
        except Exception as e:
            semantic_ok = False
            _emb_err = repr(e)

    index_cfg = (
        {"dims": EMB_DIMS, "embed": embedder, "fields": ["text"]}
        if semantic_ok
        else None
    )
    store = SQLitePersistentStore(ITEMS_DB_PATH, index=index_cfg)
    return EMB_DIMS, MEM_NS, semantic_ok, store


@app.cell
def _(CP_DB_PATH, SqliteSaver, sqlite3):
    cp_conn = sqlite3.connect(CP_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()
    return (checkpointer,)


@app.cell
def _(
    ChatNVIDIA,
    FALLBACK_MODEL,
    NVIDIA_KEY_PRESENT,
    NVIDIA_MODEL,
    NVIDIA_PKG,
    TEMPERATURE,
):
    llm = fallback_llm = None
    if NVIDIA_PKG and NVIDIA_KEY_PRESENT:
        llm = ChatNVIDIA(model=NVIDIA_MODEL, temperature=TEMPERATURE)
        fallback_llm = ChatNVIDIA(model=FALLBACK_MODEL, temperature=TEMPERATURE)
    return fallback_llm, llm


@app.cell
def _(MEM_NS, semantic_ok, store):
    def _latest_user_text(messages):
        for m in reversed(messages):
            if m.__class__.__name__ == "HumanMessage":
                return (
                    m.content if isinstance(m.content, str) else str(m.content)
                )
        return ""

    def _blend(query, k_sem=3, k_recent=2):
        """'Both': semántica + recencia, deduplicado, orden estable."""
        texts, seen = [], set()

        def add(t):
            if t and t not in seen:
                seen.add(t)
                texts.append(t)

        if semantic_ok:
            try:
                for it in store.search(MEM_NS, query=query, limit=k_sem):
                    add(it.value.get("text", ""))
            except Exception:
                pass
        for v in store.recent(MEM_NS, k_recent):
            add(v.get("text", ""))
        return texts

    return


@app.cell
def _(MEM_NS, datetime, semantic_ok, store, tool, uuid):
    def recordar(texto: str, categoria: str = "general") -> str:
        """Guarda un recuerdo DURADERO sobre la persona (preferencias, hechos,
        proyectos, decisiones). Úsalo cuando aprendas algo que valga la pena
        recordar en futuras conversaciones. `categoria` agrupa el recuerdo."""
        key = uuid.uuid4().hex[:12]
        store.put(
            MEM_NS,
            key,
            {
                "text": texto,
                "kind": categoria,
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        )
        return f"🔒 Recuerdo sellado [{categoria}]: {texto}"

    @tool
    def evocar(consulta: str, modo: str = "semantica", limite: int = 5) -> str:
        """Busca en la memoria a largo plazo. `modo`: 'semantica' (por significado),
        'palabra_clave' (coincidencia textual) o 'reciente' (lo último guardado)."""
        if modo == "reciente":
            vals = store.recent(MEM_NS, limite)
        elif modo == "palabra_clave":
            vals = store.keyword(MEM_NS, consulta, limite)
        else:
            if not semantic_ok:
                return (
                    "(Búsqueda semántica desactivada: falta NVIDIA_API_KEY. "
                    "Prueba modo='palabra_clave' o 'reciente'.)"
                )
            vals = [
                it.value
                for it in store.search(MEM_NS, query=consulta, limit=limite)
            ]
        if not vals:
            return "No encontré recuerdos relevantes."
        return "\n".join(
            f"• [{v.get('kind', '?')}] {v.get('text', '')}" for v in vals
        )

    @tool
    def olvidar(consulta: str) -> str:
        """Borra el recuerdo que mejor coincida semánticamente con `consulta`.
        Úsalo si la persona pide olvidar algo o corrige un dato viejo."""
        if not semantic_ok:
            hits = store.keyword(MEM_NS, consulta, 1)
            if not hits:
                return "No encontré nada que olvidar."
            # keyword no da key; recorremos para localizarla
            for key, it in list(store._data.get(MEM_NS, {}).items()):
                if it.value.get("text") == hits[0].get("text"):
                    store.delete(MEM_NS, key)
                    return f"🗑️ Olvidado: {hits[0].get('text', '')}"
            return "No pude localizar la clave del recuerdo."
        res = store.search(MEM_NS, query=consulta, limit=1)
        if not res:
            return "No encontré nada que olvidar."
        store.delete(MEM_NS, res[0].key)
        return f"🗑️ Olvidado: {res[0].value.get('text', '')}"

    tools = [recordar, evocar, olvidar]
    return (tools,)


@app.cell
def _(os, tool, tools):
    def search(query: str) -> str:
        """
        Herramienta de busqueda en internet basada en la API de Tavily. Devuelve resultados relevantes para la consulta.
        param: query - La consulta de busqueda.
        return: results - Resultados de busqueda en formato de texto.
        """
        try:
            from tavily import TavilyClient

            client = TavilyClient(os.getenv("TAVILY_API_KEY"))
            results = client.search(
                query, max_results=5, language="en", region="us"
            )
            return results
        except Exception as e:
            return f"Error al realizar la busqueda: {e}"

    @tool
    def research(query: str) -> str:
        """
        Herramienta de investigacion en internet basada en la API de Tavily. Devuelve resultados relevantes para la consulta.
        param: query - La consulta de investigacion.
        return: results - Resultados de investigacion en formato de texto.
        """

        try:
            from tavily import TavilyClient

            client = TavilyClient(os.getenv("TAVILY_API_KEY"))
            results = client.research(query)
            return results
        except Exception as e:
            return f"Error al realizar la investigacion: {e}"

    @tool
    def extract_webpage(url: str) -> str:
        """
        Herramienta para extraer el contenido de una pagina web basada en la API de Tavily. Devuelve el contenido relevante de la pagina.
        param: url - La URL de la pagina web.
        return: content - Contenido de la pagina web en formato de texto.
        """
        try:
            from tavily import TavilyClient

            client = TavilyClient(os.getenv("TAVILY_API_KEY"))
            content = client.extract(url)
            return content
        except Exception as e:
            return f"Error al extraer el contenido de la pagina web: {e}"

    # Listado final de tools
    tools.extend([search, research, extract_webpage])
    return


@app.cell
def _(USER_ID, dynamic_prompt):
    PERSONA = (
        "Eres el Cerebro en el Frasco: un asistente lúcido, directo y con memoria "
        "persistente. Hablas con naturalidad (español o inglés según fluya)."
    )

    @dynamic_prompt
    def inyectar_memoria(request) -> str:
        consulta = _latest_user_text(request.messages) or USER_ID
        recuerdos = _blend(consulta)
        if recuerdos:
            bloque = "\n".join(f"- {t}" for t in recuerdos)
            return (
                f"{PERSONA}\n\n## Lo que recuerdas de {USER_ID} (memoria persistente):\n"
                f"{bloque}\n\nUsa estos recuerdos con naturalidad, sin anunciarlos. "
                "Si aprendes algo nuevo y duradero, llama a `recordar`."
            )
        return (
            f"{PERSONA}\n\nAún no tienes recuerdos de esta persona. "
            "Cuando aprendas algo duradero, llama a `recordar`."
        )

    return (inyectar_memoria,)


@app.cell
def _(
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    LLMToolSelectorMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    fallback_llm,
    inyectar_memoria,
    llm,
    middleware_menu,
):
    middleware = [inyectar_memoria]  # primero => capa más externa
    enabled = [
        "inyectar_memoria (custom)",
    ]
    T = middleware_menu.value
    if llm is not None:
        if T["summarization"]:
            middleware.append(
                SummarizationMiddleware(
                    model=llm, trigger=("messages", 40), keep=("messages", 20)
                )
            )
            enabled.append("summarization")
        if T["context_editing"]:
            middleware.append(ContextEditingMiddleware())
            enabled.append("context_editing")
        if T["human_in_loop"]:
            (
                middleware.append(
                    HumanInTheLoopMiddleware(
                        interrupt_on={
                            "write_file": True,  # Todas las acciones de escritura requieren aprobación
                            "execute_sql": {
                                "allowed_decisions": ["approve", "reject"]
                            },  # Ejemplo de acción específica que requiere aprobación
                            "search": {
                                "allowed_decisions": ["approve", "reject"]
                            },  # Ejemplo de acción específica que requiere aprobación
                            "research": {
                                "allowed_decisions": ["approve", "reject"]
                            },  # Ejemplo de acción específica que requiere aprobación"
                            "extract_webpage": {
                                "allowed_decisions": ["approve", "reject"]
                            },  # Ejemplo de acción específica que requiere aprobación"
                            "read_data": False,  # Operacion segura, no requiere aprobación
                        },
                        # Prefijo de descripción para la interrupción, útil para el registro y la interfaz de usuario. necesita la acción y el contenido de la solicitud para ser informativo.
                        # e.g., "Ejecucion de herramientas debería ser aprobada: write_file, execute_sql"
                        # Herramientas individuales pueden tener su propia descripción de prefijo si se requiere.
                        description_prefix="Tool execution pending approval",
                    )
                ),
            )
        if T["model_call_limit"]:
            middleware.append(
                ModelCallLimitMiddleware(thread_limit=60, run_limit=25)
            )
            enabled.append("model_call_limit")
        if T["tool_call_limit"]:
            middleware.append(ToolCallLimitMiddleware(thread_limit=80))
            enabled.append("tool_call_limit")
        if T["tool_retry"]:
            middleware.append(ToolRetryMiddleware(max_retries=2))
            enabled.append("tool_retry")
        if T["model_retry"]:
            middleware.append(ModelRetryMiddleware(max_retries=2))
            enabled.append("model_retry")
        if T["model_fallback"] and fallback_llm is not None:
            middleware.append(ModelFallbackMiddleware(fallback_llm))
            enabled.append("model_fallback")
        if T["todo_planning"]:
            middleware.append(TodoListMiddleware())
            enabled.append("todo_planning")
        if T["tool_selector"]:
            middleware.append(LLMToolSelectorMiddleware(model=llm, max_tools=3))
            enabled.append("tool_selector")
        if T["pii"]:
            middleware.append(
                PIIMiddleware(
                    "email",
                    strategy="redact",
                    apply_to_tool_results=True,
                    apply_to_input=True,
                )
            )
            middleware.append(PIIMiddleware("url", strategy="redact"))
            middleware.append(PIIMiddleware("ip", strategy="mask"))
            middleware.append(
                PIIMiddleware(
                    "phone",
                    detector=[r"[0-9]{3}-[0-9]{3}-[0-9]{4}", r"[0-9]{10}"],
                    strategy="mask",
                    apply_to_input=True,
                    apply_to_tool_results=True,
                )
            )
            middleware.append(
                PIIMiddleware(
                    "ssn",
                    detector=[r"\d{3}-\d{2}-\d{4}"],
                    strategy="mask",
                )
            )
            enabled.append("pii(email,url)")
    return enabled, middleware


@app.cell
def _(checkpointer, create_agent, llm, middleware, store, tools, uuid):
    agent = None
    if llm is not None:
        agent = create_agent(
            model=llm,
            tools=tools,
            middleware=middleware,
            checkpointer=checkpointer,  # corto plazo en disco
            store=store,  # largo plazo en disco
        )

    # hilo nuevo por arranque (chat visible empieza limpio); la memoria a LARGO
    # plazo persiste entre sesiones -> es ahí donde ocurre el "recuerdo entre chats".
    THREAD_ID = "sesion-" + uuid.uuid4().hex[:8]
    return THREAD_ID, agent


@app.cell
def _(
    MEM_NS,
    THREAD_ID,
    USER_ID,
    agent,
    datetime,
    fallback_llm,
    llm,
    semantic_ok,
    store,
    uuid,
):
    from pydantic import BaseModel, Field
    from typing import List, Literal, Optional

    class MemoryOp(BaseModel):
        action: Literal["add", "update", "delete"] = Field(
            description="Acción a realizar: add (agregar), update (actualizar), delete (olvidar)"
        )
        content: str = Field(description="Contenido del recuerdo a guardar o actualizar")
        kind: str = Field(description="Categoría del recuerdo (ej. 'preferencia', 'hecho', 'proyecto')")
        old_content_query: Optional[str] = Field(
            description="Si es update o delete, la frase clave para encontrar el recuerdo anterior", 
            default=None
        )

    class ReflectionOutput(BaseModel):
        operations: List[MemoryOp] = Field(description="Operaciones de memoria a aplicar")

    def _reflexion_autonoma(user_text: str, agent_text: str) -> list[str]:
        """
        Capa 3 de Memoria: Reflexión Autónoma.
        Analiza el turno de conversación y decide automáticamente si hay nueva información
        relevante que deba agregarse, actualizarse o eliminarse de la memoria a largo plazo,
        sin que el usuario lo solicite explícitamente y sin necesidad de herramientas manuales.
        """
        analista = fallback_llm if fallback_llm else llm
        if not analista:
            return []

        prompt = f"""
        Analiza la siguiente interacción.
        Extrae nueva información personal duradera sobre el usuario que deba ser recordada (preferencias, contexto, hechos, proyectos).
        Si la información actualiza o contradice algo que el usuario dijo antes, genera 'update' o 'delete'. Si es nueva, 'add'.
        Si no hay información relevante para recordar a largo plazo, devuelve una lista vacía en operations.

        Usuario: {user_text}
        Asistente: {agent_text}
        """
        cambios = []
        try:
            # Utilizamos with_structured_output para que el LLM devuelva un JSON validado por Pydantic
            structured_llm = analista.with_structured_output(ReflectionOutput)
            result = structured_llm.invoke(prompt)

            for op in result.operations:
                if op.action == "add":
                    key = uuid.uuid4().hex[:12]
                    store.put(
                        MEM_NS, 
                        key, 
                        {"text": op.content, "kind": op.kind, "ts": datetime.datetime.now().isoformat(timespec="seconds")}
                    )
                    cambios.append(f"✅ **Añadido:** {op.content} *(Categoría: {op.kind})*")

                elif op.action in ["update", "delete"] and op.old_content_query:
                    hits = []
                    if semantic_ok:
                        hits = store.search(MEM_NS, query=op.old_content_query, limit=1)
                    else:
                        kw_hits = store.keyword(MEM_NS, op.old_content_query, 1)
                        if kw_hits:
                            for k, it in list(store._data.get(MEM_NS, {}).items()):
                                if it.value.get("text") == kw_hits[0].get("text"):
                                    hits = [it]
                                    break

                    if hits:
                        old_key = hits[0].key
                        old_text = hits[0].value.get("text", "")
                        store.delete(MEM_NS, old_key)
                        if op.action == "update":
                            key = uuid.uuid4().hex[:12]
                            store.put(
                                MEM_NS, 
                                key, 
                                {"text": op.content, "kind": op.kind, "ts": datetime.datetime.now().isoformat(timespec="seconds")}
                            )
                            cambios.append(f"🔄 **Actualizado:** *'{old_text}'* ➡️ *'{op.content}'*")
                        else:
                            cambios.append(f"🗑️ **Olvidado:** *'{old_text}'*")
                    else:
                        if op.action == "update":
                            key = uuid.uuid4().hex[:12]
                            store.put(
                                MEM_NS, 
                                key, 
                                {"text": op.content, "kind": op.kind, "ts": datetime.datetime.now().isoformat(timespec="seconds")}
                            )
                            cambios.append(f"✅ **Añadido (Update):** {op.content}")
        except Exception:
            # Los errores de reflexión son silenciosos para no romper el flujo del chat principal
            pass

        return cambios

    def run_agent(messages, config=None):
        if agent is None:
            yield (
                "⚠️ Falta `NVIDIA_API_KEY` (o el paquete "
                "`langchain-nvidia-ai-endpoints`). El chat está inactivo, "
                "pero la memoria a largo plazo ya está lista en disco."
            )
            return

        try:
            user_text = messages[-1].content
            cfg = {"configurable": {"thread_id": THREAD_ID, "user_id": USER_ID}}

            # 1. Ejecución principal del agente (Middlewares activos, invoca herramientas si es necesario)
            out = agent.invoke(
                {"messages": [{"role": "user", "content": user_text}]}, cfg
            )
            content = out["messages"][-1].content
            content_str = content if isinstance(content, str) else str(content)

            # 2. Mostrar respuesta inicial y aviso interactivo (aprovechando yield en Marimo UI)
            yield content_str + "\n\n*(🧠 Analizando la conversación para actualizar la memoria de forma autónoma...)*"

            # 3. Ejecutar Reflexión Autónoma (Capa 3) post-turno sin herramientas explícitas
            cambios = _reflexion_autonoma(user_text, content_str)

            # 4. Finalizar la ejecución mostrando las actualizaciones a la UI si existieron (Explicabilidad)
            if cambios:
                explicacion_cambios = "\n".join(cambios)
                yield content_str + f"\n\n---\n**🧠 Memoria Autónoma Actualizada:**\n{explicacion_cambios}"
            else:
                yield content_str

        except Exception as e:
            yield f"❌ Error al invocar el agente:\n```\n{e!r}\n```"

    return (run_agent,)


@app.cell
def _(
    CP_DB_PATH,
    EMB_DIMS,
    EMB_MODEL,
    FALLBACK_MODEL,
    ITEMS_DB_PATH,
    NVIDIA_KEY_PRESENT,
    NVIDIA_MODEL,
    TEMPERATURE,
    THREAD_ID,
    USER_ID,
    enabled,
    mo,
    semantic_ok,
    store,
):
    n_mem = sum(len(v) for v in store._data.values())

    # Indicadores de estado dinámicos
    api_status = (
        "🟢 Activa y Verificada"
        if NVIDIA_KEY_PRESENT
        else "🔴 Faltante (Chat inactivo)"
    )
    sem_status = (
        f"🟢 Activa ({EMB_DIMS}-dim)"
        if semantic_ok
        else "🔴 Desactivada (Fallo en API/Modelo)"
    )

    # Diccionario para explicar dinámicamente los middlewares activos
    mw_descriptions = {
        "inyectar_memoria (custom)": "Capa base (siempre activa) que inyecta contexto y recuerdos en el system prompt.",
        "summarization": "Condensa dinámicamente historiales de chat largos para evitar desbordar la ventana de contexto.",
        "context_editing": "Limpia y poda automáticamente resultados antiguos de herramientas (ej. búsquedas viejas).",
        "human_in_loop": "Intercepta la ejecución para solicitar tu aprobación antes de realizar acciones críticas de escritura o búsqueda.",
        "model_call_limit": "Mecanismo de seguridad que previene bucles infinitos limitando las llamadas máximas al LLM.",
        "tool_call_limit": "Pone un tope a la cantidad de herramientas que el agente puede invocar en un solo turno.",
        "tool_retry": "Captura excepciones de herramientas y obliga al modelo a reintentar con parámetros corregidos.",
        "model_retry": "Maneja errores transitorios (ej. de red o timeouts) reintentando la llamada a NVIDIA NIM.",
        "model_fallback": f"Redirige automáticamente el tráfico al modelo secundario (`{FALLBACK_MODEL}`) si el principal falla.",
        "todo_planning": "Fuerza al modelo a estructurar tareas complejas en listas de pasos a seguir antes de actuar.",
        "tool_selector": "Filtra dinámicamente las herramientas, entregando al LLM solo las más relevantes para la consulta actual.",
        "pii(email,url)": "Detecta y enmascara o redacta información personal sensible antes de que salga hacia el LLM o herramientas.",
        "pii(email,url,ip,phone,ssn)": "Detecta y enmascara o redacta información personal sensible antes de que salga hacia el LLM o herramientas.",
    }

    # Generar la lista de middlewares con sus explicaciones
    active_mw_list = "\n".join(
        [
            f"- **`{mw}`**: {mw_descriptions.get(mw, 'Módulo activo de procesamiento.')}"
            for mw in enabled
        ]
    )

    # Definir el string SIN sangría y con .strip() evita que el parser de Markdown lo trate como bloque de código
    panel_text = f"""
    ## 📊 Panel de Diagnóstico y Telemetría del Agente

    **Identidad de Memoria:** `{USER_ID}` &nbsp;&nbsp;|&nbsp;&nbsp; **ID de Sesión (Corto Plazo):** `{THREAD_ID}`

    ---

    ### 🧠 Motores de Inferencia (NVIDIA NIM)
    | Componente | Configuración Activa | Estado de Conexión |
    | :--- | :--- | :--- |
    | **API Key** | Variable de Entorno `NVIDIA_API_KEY` | {api_status} |
    | **LLM Principal** | `{NVIDIA_MODEL}` | Temperatura: `{TEMPERATURE}` |
    | **LLM Respaldo** | `{FALLBACK_MODEL}` | Listo para *Failover* automático |
    | **Motor Embeddings** | `{EMB_MODEL}` | {sem_status} |

    > *Nota: El motor de embeddings calcula representaciones vectoriales en punto flotante puro para realizar búsquedas por similitud.*

    ---

    ### 🗄️ Arquitectura de Memoria Dual (Persistencia en Disco)
    *Los datos sobreviven a los reinicios del sistema, garantizando la continuidad de la identidad del agente.*

    * **Corto Plazo (Memoria de Trabajo y Contexto):**
        * **Ruta:** `{CP_DB_PATH}`
        * **Mecánica:** Almacena el hilo exacto de la conversación actual utilizando `SqliteSaver`.
    * **Largo Plazo (Cerebro Central / Hechos y Preferencias):**
        * **Ruta:** `{ITEMS_DB_PATH}`
        * **Mecánica:** Búsqueda híbrida (Semántica Cosine + Keyword Directa + Recencia).
        * **Volumen:** **{n_mem}** recuerdo(s) consolidado(s) en la base de datos.

    ---

    ### ⚙️ Pipeline de Middlewares Activos ({len(enabled)})
    *Los middlewares envuelven al agente. Interceptan, filtran y modifican los datos de entrada/salida en tiempo real.*

    {active_mw_list}
    """.strip()

    status = mo.callout(
        mo.md(panel_text),
        kind="success" if (NVIDIA_KEY_PRESENT and semantic_ok) else "danger",
    )
    return (status,)


@app.cell
def _(status):
    status
    return


@app.cell
def _(agent):
    arch = agent.get_graph().draw_mermaid()
    return (arch,)


@app.cell
def _(arch, mo):
    mo.mermaid(arch)
    return


@app.cell
def _(mo, run_agent):
    chat = mo.ui.chat(
        run_agent,
        prompts=[
            "Hola, ¿qué recuerdas de mí?",
            "Recuerda que estoy preparando un curso de agentes de IA.",
            "¿Qué te conté sobre mis cursos?",
        ],
        show_configuration_controls=True,
    )
    chat
    return


@app.cell
def _(MEM_NS, mo, store):
    _mems = store.recent(MEM_NS, 50)
    mo.vstack(
        [
            mo.md(f"### 🗄️ Memoria a largo plazo — {len(_mems)} recuerdo(s)"),
            mo.ui.table(
                [
                    {
                        "categoría": m.get("kind", "?"),
                        "recuerdo": m.get("text", ""),
                        "guardado": m.get("ts", ""),
                    }
                    for m in _mems
                ]
            )
            if _mems
            else mo.md("_(vacío — aún no hay recuerdos)_"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
