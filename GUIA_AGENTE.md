# 🧠 El Cerebro en el Frasco — Guía del Agente

Agente conversacional con **memoria persistente en disco**, construido con
**LangChain v1** (`create_agent`) + **LangGraph**, servido por **NVIDIA NIM**
y presentado como un notebook reactivo de **marimo**.

El archivo principal y corregido es **[`agent_full.py`](agent_full.py)**.

> ℹ️ El `README.md` de este repositorio describe una app de consola anterior
> (`main.py`). Esta guía cubre el agente nuevo (`agent_full.py`). Si quieres,
> renombra este archivo a `README.md`.

---

## 1. ¿Qué hace?

Un asistente que **recuerda quién eres entre conversaciones distintas**. Combina
tres capas de memoria y un pipeline de middlewares configurable en vivo.

### Arquitectura de memoria (3 capas)

| Capa | Qué guarda | Tecnología | Persiste entre chats |
| :--- | :--- | :--- | :---: |
| **1 · Corto plazo** | El hilo exacto del chat actual | `SqliteSaver` (checkpointer) | ❌ (por sesión) |
| **2 · Largo plazo** | Hechos/preferencias del usuario | `SQLitePersistentStore` (custom) | ✅ |
| **3 · Reflexión autónoma** | Extrae hechos nuevos sin que los pidas | LLM + `with_structured_output` | ✅ |

La Capa 2 usa **búsqueda híbrida**: semántica (coseno con NumPy puro) + palabra
clave (`LIKE` en SQL) + recencia. **No requiere extensiones C** (sqlite-vec/vss),
así que funciona en Windows sin compilar nada.

### Herramientas (tools) del agente

- **Memoria:** `recordar` (escribe), `evocar` (lee), `olvidar` (borra).
- **Web (vía Tavily):** `search`, `research`, `extract_webpage`.

---

## 2. Requisitos previos

1. **[uv](https://docs.astral.sh/uv/)** instalado (gestiona Python y dependencias).
   Las dependencias están declaradas *inline* en la cabecera PEP-723 de
   `agent_full.py`; `uv` las instala solo al ejecutar. No hace falta `pip install`.
2. **Python ≥ 3.11** (lo provee `uv` automáticamente).
3. **Clave de NVIDIA NIM** (`NVIDIA_API_KEY`, empieza con `nvapi-`).
   Consíguela en <https://build.nvidia.com/>.
4. *(Opcional)* **Clave de Tavily** (`TAVILY_API_KEY`) para las tools web.
   Consíguela en <https://tavily.com/>.

> **Sin `NVIDIA_API_KEY` el notebook NO se rompe**: arranca en *modo degradado*
> (el chat queda inactivo, pero la memoria en disco y los paneles funcionan).

---

## 3. Configuración: variables de entorno

Crea un archivo **`.env`** en la raíz del proyecto (junto a `agent_full.py`).
El notebook lo carga automáticamente con `python-dotenv`.

```dotenv
# --- OBLIGATORIA para activar el chat ---
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# --- OPCIONAL: tools de búsqueda web ---
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx

# --- OPCIONAL: modelos NVIDIA NIM (tienen valores por defecto) ---
NIM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
NIM_FALLBACK=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
NIM_EMBED=nvidia/nv-embedqa-e5-v5
NIM_TEMPERATURE=0.6

# --- OPCIONAL: identidad y ubicación de la memoria ---
CEREBRO_USER=leonardo
# Por defecto la memoria vive en ~/.cerebro_memory (anclada al HOME, no al cwd).
# CEREBRO_DIR=C:\ruta\personalizada\memoria
```

| Variable | Obligatoria | Default | Para qué sirve |
| :--- | :---: | :--- | :--- |
| `NVIDIA_API_KEY` | ✅ (chat) | — | Autentica el LLM y los embeddings. |
| `TAVILY_API_KEY` | ❌ | — | Habilita `search`/`research`/`extract_webpage`. |
| `NIM_MODEL` | ❌ | `nemotron-3-ultra-550b` | LLM principal. |
| `NIM_FALLBACK` | ❌ | `nemotron-3-nano-omni-30b` | LLM de respaldo (failover + reflexión). |
| `NIM_EMBED` | ❌ | `nv-embedqa-e5-v5` | Motor de embeddings (búsqueda semántica). |
| `NIM_TEMPERATURE` | ❌ | `0.6` | Creatividad del LLM (0 = determinista). |
| `CEREBRO_USER` | ❌ | `leonardo` | Namespace de la memoria de largo plazo. |
| `CEREBRO_DIR` | ❌ | `~/.cerebro_memory` | Carpeta de las bases SQLite. |

> ⚠️ **Añade `.env` a tu `.gitignore`** — nunca subas claves al repositorio.

---

## 4. Cómo ejecutarlo

Desde la raíz del proyecto:

```powershell
# Modo edición (recomendado: ves y editas todas las celdas)
uv run marimo edit agent_full.py

# Modo app (solo la interfaz, sin código a la vista)
uv run marimo run agent_full.py
```

`uv` resuelve e instala las dependencias la primera vez (puede tardar).
Se abrirá el notebook en tu navegador. Baja hasta la celda de **chat** y escribe.

### Prueba rápida

1. *"Recuerda que estoy preparando un curso de agentes de IA."* → el agente guarda el hecho.
2. Recarga / nueva sesión.
3. *"¿Qué te conté sobre mis cursos?"* → debe recordarlo (memoria de largo plazo).

---

## 5. Pipeline de middlewares

Los middlewares **envuelven** al agente (orden = de afuera hacia adentro).
Se activan/desactivan en vivo desde el menú interactivo del notebook
(celda *Agent Middleware Settings*).

| Middleware | Default | Qué hace |
| :--- | :---: | :--- |
| `inyectar_memoria` (custom) | siempre | Inyecta recuerdos relevantes en el system prompt. |
| `summarization` | ON | Resume historiales largos para no desbordar el contexto. |
| `context_editing` | ON | Poda resultados viejos de tools. |
| `human_in_loop` | **OFF** | Pide aprobación antes de tools críticas. **Pausa el grafo** (ver nota). |
| `model_call_limit` | ON | Tope de llamadas al LLM (anti bucles infinitos). |
| `tool_call_limit` | ON | Tope de llamadas a tools por turno. |
| `tool_retry` | ON | Reintenta tools que fallan. |
| `model_retry` | ON | Reintenta el LLM ante errores transitorios. |
| `model_fallback` | ON | Cae al modelo de respaldo si el principal falla. |
| `todo_planning` | ON | Estructura tareas complejas en listas de pasos. |
| `tool_selector` | OFF | Filtra tools dinámicamente (útil con MUCHAS tools). |
| `pii` | OFF | Enmascara/redacta datos personales (email, url, ip, phone, ssn). |

> 🔴 **Nota sobre `human_in_loop`:** viene **apagado por defecto**. Cuando se
> activa, `HumanInTheLoopMiddleware` **pausa el grafo** (devuelve un estado
> `__interrupt__`) esperando que apruebes/rechaces la acción. La interfaz de
> chat de marimo **no tiene un botón de aprobación** para reanudarlo, así que
> el chat quedaría congelado. `run_agent` detecta ese interrupt y muestra un
> aviso claro en vez de romperse. Es ideal para *demostrar* el patrón; para
> usarlo de verdad necesitarías una UI de aprobación dedicada.

---

## 6. Dónde se guardan los datos

Dos bases SQLite bajo `~/.cerebro_memory/` (o `CEREBRO_DIR`):

- `hilos_corto_plazo.db` → checkpoints de conversación (Capa 1).
- `memorias_largo_plazo.db` → hechos duraderos + vectores (Capa 2).
  - Tabla `store_items(namespace, key, value, created_at, updated_at)`.
  - Tabla `store_vectors(namespace, key, path, vector)`.
  - `value` es JSON: `{"text": "...", "kind": "preferencia", "ts": "2026-06-29T12:00:00"}`.

Para **empezar de cero**, borra esa carpeta (perderás todos los recuerdos).

---

## 7. Correcciones aplicadas en esta versión

`agent_full.py` corrige los errores de integración del notebook original.
Cada arreglo está comentado en el código con la etiqueta `# [FIX]`.

| # | Problema | Arreglo |
| :--: | :--- | :--- |
| 1 | Markdown del encabezado duplicado. | Una sola copia. |
| 2 | `NVIDIA_KEY_PRESENT`: `None.startswith` → crash sin API key. | `(... or "").startswith(...)`. |
| 3 | `human_in_loop` ON congelaba el chat (pausa el grafo). | Default OFF + manejo del interrupt. |
| 4 | Helpers `_blend`/`_latest_user_text` con guion bajo = locales a la celda en marimo → `NameError`. | Renombrados y exportados. |
| 5 | `recordar` y `search` sin `@tool` → no eran tools válidas. | Añadido `@tool`. |
| 6 | `tools.extend(...)` entre celdas → carrera reactiva en marimo. | `tools = memory_tools + web_tools` (lista nueva). |
| 7 | `inyectar_memoria` no recibía los helpers. | Firma de celda actualizada. |
| 8 | `human_in_loop`: tupla accidental + `interrupt_on` con nombres de tools inexistentes. | Remapeado a las tools reales. |
| 9 | Diagrama Mermaid (`arch`) crasheaba si `agent is None`. | Grafo de marcador en modo degradado. |
| 10 | `run_agent` reventaba al recibir un `__interrupt__`. | Guardia que avisa con elegancia. |

Validado con `python -m py_compile` (sintaxis) y un verificador AST del grafo de
dataflow de marimo (28 celdas, 64 símbolos, sin referencias colgantes ni
definiciones duplicadas).

---

## 8. Problemas comunes

| Síntoma | Causa probable | Solución |
| :--- | :--- | :--- |
| `🔴 API Key Faltante` en el panel | No hay `NVIDIA_API_KEY` o no empieza con `nvapi`. | Revisa el `.env`. |
| Búsqueda semántica desactivada | Falla la API de embeddings o falta la key. | Usa `evocar` con `modo='palabra_clave'` o `'reciente'`. |
| Las tools web devuelven error | Falta `TAVILY_API_KEY`. | Añádela al `.env`. |
| El chat se queda "pensando" | `human_in_loop` activado. | Desactívalo en el menú de middlewares. |
| "No me guardó nada" | Carpeta de memoria distinta entre ejecuciones. | Fija `CEREBRO_DIR` a una ruta absoluta. |
