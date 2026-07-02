# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo[mcp]",
#     "langchain>=1.0",
#     "langgraph>=1.0",
#     "langgraph-checkpoint-sqlite",
#     "langchain-nvidia-ai-endpoints",
#     "python-dotenv",
#     "tavily-python",
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
#     "deepagents>=0.2",
#     "pyyaml>=6",
# ]
# ///
#
# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  AGENTE PROFUNDO · DeepAgents + Skills + Subagentes + Multimodal                     ║
# ║  Material Educativo: Construcción de Agentes con LangChain / LangGraph               ║
# ║  LangChain / LangGraph + Memoria Persistente + Middlewares + Marimo UI               ║
# ║                                                                                      ║
# ║  Ejecutar:  uv run marimo edit agent_deep.py                                         ║
# ║  Requiere:  NVIDIA_API_KEY  en .env  (export / $env:NVIDIA_API_KEY="nvapi-...")       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# CONCEPTO 1 · ¿QUÉ ES UN AGENTE DE IA?
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  Un "agente" es un programa que combina un LLM con herramientas (tools) externas.
#  En lugar de generar solo texto, el agente puede:
#
#    1. RAZONAR  → el LLM decide qué hacer con la petición del usuario.
#    2. ACTUAR   → llama a una herramienta (buscar en web, guardar en BD, etc.).
#    3. OBSERVAR → recibe el resultado de la herramienta.
#    4. REPETIR  → si necesita más información, vuelve al paso 1.
#    5. RESPONDER → cuando tiene suficiente contexto, genera la respuesta final.
#
#  Este ciclo se llama ReAct (Reason + Act). LangGraph implementa el grafo de
#  estados que lo orquesta; LangChain provee las abstracciones de alto nivel.
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# CONCEPTO 2 · ARQUITECTURA DE MEMORIA DUAL
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  CAPA 1 · Corto Plazo  (SqliteSaver — "checkpointer")
#    ● Persiste el HILO de conversación turno a turno en SQLite.
#    ● El agente puede leer qué se dijo 20 turnos atrás en la misma sesión.
#    ● Cada sesión tiene un THREAD_ID único; al cambiar de hilo, el historial
#      de mensajes empieza vacío (pero la memoria de largo plazo permanece).
#
#  CAPA 2 · Largo Plazo  (AlmacenPersistenteSQLite — "store")
#    ● Almacena HECHOS DURABLES del usuario: preferencias, proyectos, gustos.
#    ● Persiste entre TODAS las sesiones en el mismo archivo .db.
#    ● Búsqueda HÍBRIDA (las tres disponibles simultáneamente):
#        · Semántica   → similitud coseno entre embeddings de NVIDIA.
#        · Keyword     → SQL LIKE sobre el valor JSON serializado.
#        · Recencia    → ORDER BY updated_at DESC.
#    ● Sin extensiones C (sqlite-vec / vss) → funciona en Windows sin compilar.
#
#  CAPA 3 · Reflexión Autónoma  (ejecutar_reflexion_autonoma)
#    ● Después de cada turno, el modelo de respaldo analiza la conversación.
#    ● Extrae hechos nuevos y los graba en CAPA 2 sin que el usuario lo pida.
#    ● Si detecta información desactualizada, puede actualizarla o eliminarla.
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# CONCEPTO 3 · HERRAMIENTAS (TOOLS)
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  Las herramientas son funciones Python que el agente puede invocar de forma
#  autónoma cuando lo necesita. Se declaran con el decorador @tool de LangChain,
#  que construye automáticamente:
#    · nombre      → identificador que el LLM recibe.
#    · descripción → docstring Python, que el LLM usa para decidir cuándo invocarla.
#    · schema      → inferido de los type hints; el LLM lo usa para los argumentos.
#
#  TOOLS DE MEMORIA (operan sobre CAPA 2):
#    · recordar(texto, categoria)          → graba un hecho duradero.
#    · evocar(consulta, modo, limite)      → busca recuerdos.
#    · olvidar(consulta)                   → borra el recuerdo más similar.
#
#  TOOLS WEB (requieren TAVILY_API_KEY en .env):
#    · buscar_en_red(consulta)             → búsqueda rápida en internet.
#    · investigar_a_fondo(consulta)        → investigación profunda multi-fuente.
#    · extraer_pagina_web(url)             → extrae texto de una URL concreta.
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# CONCEPTO 4 · MIDDLEWARES
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  Los middlewares son capas de procesamiento que ENVUELVEN al agente.
#  Se aplican en el orden de la lista: el primero = capa más externa (corre primero).
#  Cada capa puede modificar la entrada, la salida, o interrumpir la ejecución.
#
#  PIPELINE (exterior → interior):
#    inyectar_memoria_dinamica   ← @dynamic_prompt custom; inyecta recuerdos en el system prompt
#    SummarizationMiddleware     ← resume histórico cuando supera N mensajes
#    ContextEditingMiddleware    ← poda resultados de tools para liberar contexto
#    HumanInTheLoopMiddleware    ← pausa y pide aprobación antes de acciones críticas
#    ModelCallLimitMiddleware    ← tope de seguridad en llamadas al LLM
#    ToolCallLimitMiddleware     ← tope de seguridad en llamadas a tools
#    ToolRetryMiddleware         ← reintenta herramientas que lanzan excepción
#    ModelRetryMiddleware        ← reintenta el LLM ante errores de red/timeout
#    ModelFallbackMiddleware     ← redirecciona al modelo de respaldo si el principal falla
#    TodoListMiddleware          ← fuerza al LLM a planificar antes de actuar
#    LLMToolSelectorMiddleware   ← filtra tools dinámicamente (útil con >10 tools)
#    PIIMiddleware               ← detecta y enmascara datos personales sensibles
#
# ─────────────────────────────────────────────────────────────────────────────────────
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# CONCEPTO 5 · DEEP AGENTS
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  create_deep_agent() envuelve create_agent() y añade de serie:
#    · Planificación   → tool write_todos integrada (NO añadir TodoListMiddleware).
#    · Filesystem      → ls / read_file / write_file / edit_file sobre un backend.
#    · Subagentes      → tool `task` que delega en personajes con contexto limpio.
#    · Skills          → carga SKILL.md con "progressive disclosure": al inicio solo
#                        ve los frontmatter; lee el contenido completo bajo demanda.
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# CONCEPTO 6 · AGENT SKILLS (estándar agentskills.io)
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  Una skill es una carpeta con un SKILL.md (frontmatter YAML: name, description +
#  instrucciones markdown) y recursos opcionales. Es EL MISMO formato que usan
#  Claude Code y otros arneses → las skills del marketplace son reutilizables aquí.
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# CONCEPTO 7 · SUBAGENTES COMO REPARTO DE UNA OBRA
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  Cada personaje vive en ./subagentes/<nombre>.md (frontmatter + persona).
#  El agente principal (el "director") lee las descriptions y delega escenas
#  vía la tool `task`. El personaje trabaja con contexto propio y devuelve
#  solo su resultado — el contexto del director no se contamina.
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# CONCEPTO 8 · SALIDAS MULTIMODALES
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  Convención: toda tool que produzca algo no-textual lo guarda en ./artefactos/
#  y devuelve la ruta. La Galería renderiza por tipo (imagen, pdf, video, audio,
#  tabla…) y el chat muestra inline los artefactos creados en el turno.

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧠 Agente Profundo — DeepAgents
    ## Material Educativo: DeepAgents + Skills + Subagentes + Multimodal

    Este notebook es una guía práctica completa para entender y construir un **agente
    de IA con memoria persistente** usando herramientas modernas del ecosistema Python.

    ### ¿Qué vas a aprender?

    | Módulo | Concepto |
    | :--- | :--- |
    | **Agentes ReAct** | Ciclo Razonar → Actuar → Observar en LangGraph |
    | **Memoria Dual** | Corto plazo (hilo) + Largo plazo (hechos duraderos) en SQLite |
    | **Tools** | Cómo declarar herramientas que el LLM puede invocar autónomamente |
    | **Middlewares** | Capas de interceptación que modifican el comportamiento del agente |
    | **Reflexión Autónoma** | El agente extrae y actualiza hechos sin que el usuario lo pida |
    | **Skills** | Carpetas SKILL.md con progressive disclosure (estándar agentskills.io) |
    | **Subagentes** | Personajes con contexto propio delegados vía la tool `task` |
    | **Multimodal** | Artefactos no-textuales guardados en ./artefactos/ y renderizados en galería |
    | **Marimo UI** | Dashboard reactivo, panel de control, chat, inspector de memoria |

    > **Requisito:** Define `NVIDIA_API_KEY=nvapi-...` en tu archivo `.env` o como
    > variable de entorno. Opcionalmente, `TAVILY_API_KEY` para las búsquedas web.

    ---
    """)
    return


@app.cell
def _():
    import os
    import sqlite3
    import json
    import uuid
    import datetime
    import numpy as np
    from pathlib import Path

    # ── Utilidades ────────────────────────────────────────────────────────────────────
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    from typing import List, Literal, Optional

    # ── Núcleo LangChain / LangGraph ──────────────────────────────────────────────────
    from langchain.tools import tool
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
        PIIMiddleware,
    )
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.store.base import Item, PutOp

    # ── Integración NVIDIA (opcional) ────────────────────────────────────────────────
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

        PAQUETE_NVIDIA = True
    except ImportError:
        PAQUETE_NVIDIA = False
        ChatNVIDIA = NVIDIAEmbeddings = None

    # ── DeepAgents ────────────────────────────────────────────────────────────────────
    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend

    # ── Módulo de soporte local (funciones puras testeadas con pytest) ───────────────
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import deep_soporte as ds

    # Cargar variables de entorno desde .env (NVIDIA_API_KEY, TAVILY_API_KEY…)
    load_dotenv()
    return (
        BaseModel,
        ChatNVIDIA,
        ContextEditingMiddleware,
        Field,
        FilesystemBackend,
        HumanInTheLoopMiddleware,
        InMemoryStore,
        Item,
        LLMToolSelectorMiddleware,
        List,
        Literal,
        ModelCallLimitMiddleware,
        ModelFallbackMiddleware,
        ModelRetryMiddleware,
        NVIDIAEmbeddings,
        Optional,
        PAQUETE_NVIDIA,
        PIIMiddleware,
        Path,
        PutOp,
        SqliteSaver,
        SummarizationMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
        create_deep_agent,
        datetime,
        ds,
        dynamic_prompt,
        json,
        np,
        os,
        sqlite3,
        sys,
        tool,
        uuid,
    )


@app.cell
def _(Path, ds, os):
    PRESENCIA_API_NVIDIA = bool(
        os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi")
    )

    # ── Modelos NVIDIA NIM ────────────────────────────────────────────────────────────
    MODELO_ESTANDAR = os.environ.get(
        "NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
    )
    MODELO_RAZONAMIENTO = os.environ.get(
        "NIM_FALLBACK", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    )
    MODELO_EMBEDDINGS = os.environ.get("NIM_EMBED", "nvidia/nv-embedqa-e5-v5")

    # ── Identidad del usuario y rutas de persistencia ────────────────────────────────
    ID_USUARIO = os.environ.get("CEREBRO_USER", "usuario_principal")

    DIRECTORIO_MEMORIA = Path(
        os.environ.get("CEREBRO_DIR", Path.home() / ".memoria_cerebro")
    )
    DIRECTORIO_MEMORIA.mkdir(parents=True, exist_ok=True)

    # Ruta absoluta → nunca depende del cwd (causa #1 de "no me guardó nada" en Windows)
    RUTA_BD_LARGO_PLAZO = str(
        (DIRECTORIO_MEMORIA / "memorias_largo_plazo.db").resolve()
    )
    RUTA_BD_CORTO_PLAZO = str(
        (DIRECTORIO_MEMORIA / "hilos_corto_plazo.db").resolve()
    )

    # ── Raíz del proyecto y directorios del agente profundo ─────────────────────────
    RAIZ_PROYECTO = Path(__file__).parent.resolve()
    DIR_SKILLS = RAIZ_PROYECTO / "skills"
    DIR_SUBAGENTES = RAIZ_PROYECTO / "subagentes"
    DIR_ARTEFACTOS = RAIZ_PROYECTO / "artefactos"
    for _d in (DIR_SKILLS, DIR_SUBAGENTES, DIR_ARTEFACTOS):
        _d.mkdir(parents=True, exist_ok=True)

    # Sembrar ejemplos didácticos la primera vez (idempotente)
    ds.sembrar_skill_ejemplo(DIR_SKILLS)
    ds.sembrar_subagente_ejemplo(DIR_SUBAGENTES)
    return (
        DIR_ARTEFACTOS,
        DIR_SKILLS,
        DIR_SUBAGENTES,
        ID_USUARIO,
        MODELO_EMBEDDINGS,
        MODELO_ESTANDAR,
        MODELO_RAZONAMIENTO,
        PRESENCIA_API_NVIDIA,
        RAIZ_PROYECTO,
        RUTA_BD_CORTO_PLAZO,
        RUTA_BD_LARGO_PLAZO,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ⚙️ Panel de Control del Agente

    Ajusta los parámetros del LLM y activa/desactiva middlewares.
    Marimo reconstruye el agente y el diagrama de arquitectura **en tiempo real**
    cada vez que cambias un valor — sin necesidad de reiniciar nada.
    """)
    return


@app.cell
def _(mo):
    ui_temperatura = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.6,
        label="**Temperatura** — Creatividad / aleatoriedad del modelo",
    )
    ui_top_p = mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.05,
        value=0.95,
        label="**Top-P** — Diversidad del vocabulario (nucleus sampling)",
    )
    ui_max_tokens = mo.ui.slider(
        start=256,
        stop=65536,
        step=256,
        value=16384,
        label="**Máximo de tokens** — Longitud máxima de la respuesta",
    )
    ui_reason_budget = mo.ui.slider(
        start=1024,
        stop=32768,
        step=256,
        value=8192,
        label="**Tokens de pensamiento** — Presupuesto para razonamiento interno (CoT)",
    )
    ui_razonamiento = mo.ui.switch(
        value=True,
        label="🧠 **Habilitar Razonamiento (Chain-of-Thought)** — Usa el modelo de razonamiento",
    )
    return (
        ui_max_tokens,
        ui_razonamiento,
        ui_reason_budget,
        ui_temperatura,
        ui_top_p,
    )


@app.cell
def _(mo):
    menu_middlewares = mo.ui.dictionary(
        {
            "resumen_conversacion": mo.ui.switch(
                value=True,
                label="📄 **Resumen Automático** — Condensa historiales largos para no agotar el contexto",
            ),
            "modelo_respaldo": mo.ui.switch(
                value=True,
                label="🔄 **Modelo de Respaldo (Fallback)** — Redirige al modelo secundario si el principal falla",
            ),
            "edicion_contexto": mo.ui.switch(
                value=True,
                label="✂️ **Edición de Contexto** — Poda resultados viejos de herramientas para liberar tokens",
            ),
            "humano_en_bucle": mo.ui.switch(
                value=False,
                label="🙋 **Humano en el Bucle** — Pausa y pide aprobación antes de acciones críticas",
            ),
            "limite_llamadas_modelo": mo.ui.switch(
                value=True,
                label="🛑 **Límite: Llamadas al Modelo** — Previene bucles infinitos (máx. 60/hilo, 25/run)",
            ),
            "limite_llamadas_herramienta": mo.ui.switch(
                value=True,
                label="🛑 **Límite: Herramientas** — Tope de seguridad en invocaciones de tools (máx. 80/hilo)",
            ),
            "reintento_herramienta": mo.ui.switch(
                value=True,
                label="♻️ **Reintento de Herramienta** — Reintenta tools que lanzan excepción (máx. 2 veces)",
            ),
            "reintento_modelo": mo.ui.switch(
                value=True,
                label="♻️ **Reintento del Modelo** — Reintenta el LLM ante errores de red o timeouts",
            ),
            "filesystem_protegido": mo.ui.switch(
                value=False,
                label="📁 **Filesystem Protegido** — Pide aprobación antes de write_file / edit_file",
            ),
            "selector_herramientas": mo.ui.switch(
                value=False,
                label="🔍 **Selector Dinámico de Herramientas** — Filtra tools por relevancia (útil con >10 tools)",
            ),
            "censura_datos_personales": mo.ui.switch(
                value=False,
                label="🔒 **Censura PII** — Detecta y enmascara email, URL, IP, teléfono y NSS",
            ),
        }
    )
    return (menu_middlewares,)


@app.cell
def _(
    menu_middlewares,
    mo,
    ui_max_tokens,
    ui_razonamiento,
    ui_reason_budget,
    ui_temperatura,
    ui_top_p,
):
    _panel_llm = mo.md(f"""
    ### 🎛️ Parámetros del Motor de Inferencia
    *Controlan directamente cómo genera texto el LLM.*

    {ui_temperatura}

    {ui_top_p}

    {ui_max_tokens}

    {ui_reason_budget}

    {ui_razonamiento}
    """)

    _panel_mw = mo.md(f"""
    ### 🛡️ Pipeline de Middlewares
    *Activa o desactiva capas de procesamiento. El diagrama de arquitectura
    se actualiza en tiempo real según tu selección.*

    {menu_middlewares}
    """)

    mo.hstack([_panel_llm, _panel_mw], widths=[1, 1], gap=2)
    return


@app.cell
def _(InMemoryStore, Item, PutOp, json, np, sqlite3):
    class AlmacenPersistenteSQLite(InMemoryStore):
        """Extiende InMemoryStore con persistencia transparente en SQLite.

        Hereda toda la lógica de búsqueda semántica (coseno, ANN) de InMemoryStore
        y añade write-through a disco para que los datos sobrevivan reinicios.
        """

        def __init__(self, ruta_bd: str, *, index=None):
            super().__init__(index=index)
            self._ruta_bd = str(ruta_bd)
            self._conexion = sqlite3.connect(
                self._ruta_bd, check_same_thread=False
            )
            self._conexion.execute("PRAGMA journal_mode=WAL;")
            self._inicializar_esquema()
            self._cargar_datos()

        def _inicializar_esquema(self):
            self._conexion.executescript("""
                CREATE TABLE IF NOT EXISTS store_items(
                    namespace      TEXT,
                    clave          TEXT,
                    valor          TEXT,
                    creado_el      TEXT,
                    actualizado_el TEXT,
                    PRIMARY KEY(namespace, clave)
                );
                CREATE TABLE IF NOT EXISTS store_vectores(
                    namespace TEXT,
                    clave     TEXT,
                    ruta      TEXT,
                    vector    BLOB,
                    PRIMARY KEY(namespace, clave, ruta)
                );
            """)
            self._conexion.commit()

        @staticmethod
        def _ns_a_str(ns: tuple) -> str:
            return json.dumps(list(ns))

        @staticmethod
        def _str_a_ns(s: str) -> tuple:
            return tuple(json.loads(s))

        def _cargar_datos(self):
            """Carga ítems y vectores desde SQLite al InMemoryStore en RAM al iniciar."""
            for (
                ns_s,
                clave,
                valor,
                creado,
                actualizado,
            ) in self._conexion.execute(
                "SELECT namespace,clave,valor,creado_el,actualizado_el FROM store_items"
            ):
                ns = self._str_a_ns(ns_s)
                self._data[ns][clave] = Item(
                    value=json.loads(valor),
                    key=clave,
                    namespace=ns,
                    created_at=creado,
                    updated_at=actualizado,
                )
            for ns_s, clave, ruta, blob in self._conexion.execute(
                "SELECT namespace,clave,ruta,vector FROM store_vectores"
            ):
                ns = self._str_a_ns(ns_s)
                self._vectors[ns][clave][ruta] = np.frombuffer(
                    blob, dtype=np.float32
                ).tolist()

        def _persistir(self, operaciones):
            """Sincroniza a SQLite las operaciones ya aplicadas en RAM."""
            c = self._conexion
            for op in operaciones:
                if not isinstance(op, PutOp):
                    continue
                ns_s = self._ns_a_str(op.namespace)

                if op.value is None:  # borrado
                    c.execute(
                        "DELETE FROM store_items WHERE namespace=? AND clave=?",
                        (ns_s, op.key),
                    )
                    c.execute(
                        "DELETE FROM store_vectores WHERE namespace=? AND clave=?",
                        (ns_s, op.key),
                    )
                    continue

                item = self._data[op.namespace][op.key]
                c.execute(
                    "INSERT OR REPLACE INTO store_items VALUES(?,?,?,?,?)",
                    (
                        ns_s,
                        op.key,
                        json.dumps(item.value),
                        str(item.created_at),
                        str(item.updated_at),
                    ),
                )
                c.execute(
                    "DELETE FROM store_vectores WHERE namespace=? AND clave=?",
                    (ns_s, op.key),
                )
                for ruta, vector in (
                    self._vectors.get(op.namespace, {}).get(op.key, {}).items()
                ):
                    c.execute(
                        "INSERT OR REPLACE INTO store_vectores VALUES(?,?,?,?)",
                        (
                            ns_s,
                            op.key,
                            ruta,
                            np.asarray(vector, dtype=np.float32).tobytes(),
                        ),
                    )
            c.commit()

        def batch(self, ops):
            ops = list(ops)
            res = super().batch(ops)  # aplica en RAM + genera embeddings
            self._persistir(ops)  # sincroniza a disco
            return res

        async def abatch(self, ops):
            ops = list(ops)
            res = await super().abatch(ops)
            self._persistir(ops)
            return res

        # ── Métodos de consulta adicionales (SQL directo, sin embeddings) ────────────

        def recientes(self, namespace: tuple, limite: int = 5) -> list[dict]:
            """Devuelve los `limite` ítems más recientemente actualizados."""
            filas = self._conexion.execute(
                "SELECT valor FROM store_items "
                "WHERE namespace=? ORDER BY actualizado_el DESC LIMIT ?",
                (self._ns_a_str(namespace), limite),
            ).fetchall()
            return [json.loads(v) for (v,) in filas]

        def palabra_clave(
            self, namespace: tuple, termino: str, limite: int = 10
        ) -> list[dict]:
            """Búsqueda LIKE sobre el valor JSON serializado (case-insensitive)."""
            filas = self._conexion.execute(
                "SELECT valor FROM store_items "
                "WHERE namespace=? AND lower(valor) LIKE ? "
                "ORDER BY actualizado_el DESC LIMIT ?",
                (self._ns_a_str(namespace), f"%{termino.lower()}%", limite),
            ).fetchall()
            return [json.loads(v) for (v,) in filas]

    return (AlmacenPersistenteSQLite,)


@app.cell
def _(
    AlmacenPersistenteSQLite,
    ID_USUARIO,
    MODELO_EMBEDDINGS,
    NVIDIAEmbeddings,
    PAQUETE_NVIDIA,
    PRESENCIA_API_NVIDIA,
    RUTA_BD_LARGO_PLAZO,
):
    ESPACIO_MEMORIA = ("memorias", ID_USUARIO)
    incrustador = None
    DIMENSIONES_EMB = None
    semantica_activa = False

    if PAQUETE_NVIDIA and PRESENCIA_API_NVIDIA:
        try:
            incrustador = NVIDIAEmbeddings(
                model=MODELO_EMBEDDINGS, truncate="END"
            )
            DIMENSIONES_EMB = len(
                incrustador.embed_query("sonda de dimensiones")
            )
            semantica_activa = True
        except Exception:
            semantica_activa = False

    configuracion_indice = (
        {"dims": DIMENSIONES_EMB, "embed": incrustador, "fields": ["text"]}
        if semantica_activa
        else None
    )
    almacen_memoria = AlmacenPersistenteSQLite(
        RUTA_BD_LARGO_PLAZO, index=configuracion_indice
    )
    return DIMENSIONES_EMB, ESPACIO_MEMORIA, almacen_memoria, semantica_activa


@app.cell
def _(RUTA_BD_CORTO_PLAZO, SqliteSaver, sqlite3):
    _conexion_cp = sqlite3.connect(RUTA_BD_CORTO_PLAZO, check_same_thread=False)
    gestor_puntos_control = SqliteSaver(_conexion_cp)
    gestor_puntos_control.setup()
    return (gestor_puntos_control,)


@app.cell
def _(
    ChatNVIDIA,
    MODELO_ESTANDAR,
    MODELO_RAZONAMIENTO,
    PAQUETE_NVIDIA,
    PRESENCIA_API_NVIDIA,
    ui_max_tokens,
    ui_razonamiento,
    ui_temperatura,
    ui_top_p,
):
    llm_principal = llm_respaldo = None
    nombre_modelo_activo = (
        MODELO_RAZONAMIENTO if ui_razonamiento.value else MODELO_ESTANDAR
    )

    if PAQUETE_NVIDIA and PRESENCIA_API_NVIDIA:
        _params = {
            "temperature": ui_temperatura.value,
            "top_p": ui_top_p.value,
            "max_tokens": ui_max_tokens.value,
        }
        _thinking = {"enable_thinking": ui_razonamiento.value}

        llm_principal = ChatNVIDIA(
            model=nombre_modelo_activo,
            **_params,
            chat_template_kwargs=_thinking,
        )
        # El respaldo siempre usa el modelo estándar (más estable)
        llm_respaldo = ChatNVIDIA(
            model=MODELO_ESTANDAR,
            **_params,
            chat_template_kwargs=_thinking,
        )
    llm_estandar_obj = llm_respaldo  # alias claro para el resolvedor de subagentes
    return llm_estandar_obj, llm_principal, llm_respaldo, nombre_modelo_activo


@app.cell
def _(ESPACIO_MEMORIA, almacen_memoria, semantica_activa):
    def ultimo_texto_usuario(mensajes: list) -> str:
        """Extrae el último mensaje del usuario de la lista del agente."""
        for m in reversed(mensajes):
            if m.__class__.__name__ == "HumanMessage":
                return (
                    m.content if isinstance(m.content, str) else str(m.content)
                )
        return ""

    def mezclar_recuerdos(
        consulta: str, k_semantica: int = 3, k_recientes: int = 2
    ) -> list[str]:
        """Recuperación híbrida: semántica + recencia, deduplicada, orden estable."""
        textos: list[str] = []
        vistos: set[str] = set()

        def agregar(t: str):
            if t and t not in vistos:
                vistos.add(t)
                textos.append(t)

        if semantica_activa:
            try:
                for item in almacen_memoria.search(
                    ESPACIO_MEMORIA, query=consulta, limit=k_semantica
                ):
                    agregar(item.value.get("text", ""))
            except Exception:
                pass  # falla graciosamente

        for v in almacen_memoria.recientes(ESPACIO_MEMORIA, k_recientes):
            agregar(v.get("text", ""))

        return textos

    return mezclar_recuerdos, ultimo_texto_usuario


if __name__ == "__main__":
    app.run()
