# Diseño: `agent_deep.py` — Agente educativo sobre DeepAgents con Skills de marketplace y salidas multimodales

**Fecha:** 2026-07-01
**Estado:** Aprobado por el usuario
**Contexto:** Evolución de `agent_full.py` (notebook Marimo educativo, LangChain/LangGraph `create_agent`, memoria dual SQLite) hacia el framework DeepAgents, conservando toda la funcionalidad existente y añadiendo soporte de Agent Skills (estándar agentskills.io / SKILL.md, el mismo que usan Claude Code y otros arneses).

## Decisiones del usuario

1. **Skills:** carga local desde `./skills/` + tool instaladora desde marketplace (repos GitHub: `anthropics/skills`, `langchain-ai/langchain-skills`, o URL directa) + panel UI en Marimo.
2. **Formato:** archivo nuevo `agent_deep.py`, notebook Marimo educativo. `agent_full.py` queda intacto como referencia de la versión LangChain pura.
3. **Extras DeepAgents:** subagentes, filesystem tools expuestas, y todo lo necesario para un agente polivalente generalista, modular y fácilmente mejorable.
4. **Multimodal:** panel de salidas multimodales (imágenes, PDF, video, audio, tablas) para artefactos producidos por tools o por el agente.
5. **Constructor de subagentes:** panel/editor para crear subagentes como "papeles de una obra" (personas), persistidos en `./subagentes/*.md` e integrados en `create_deep_agent(subagents=...)`.

## Enfoque elegido

**DeepAgents nativo + middlewares passthrough.** `create_deep_agent()` reemplaza a `create_agent()` y aporta de serie: planning (`write_todos`), filesystem tools virtuales, subagentes (`task` tool) y skills con progressive disclosure. Todo lo custom del notebook original pasa sin cambios por los parámetros `middleware=`, `store=`, `checkpointer=`.

Alternativas descartadas:
- *create_agent + skill middleware manual:* reinventa lo que DeepAgents ya provee; sin subagentes ni filesystem gratis.
- *Harness dcode (CLI):* no embebible en Marimo.

## Arquitectura

### 1 · Núcleo

```python
agente = create_deep_agent(
    model=ChatNVIDIA(...),            # mismos modelos NIM y sliders que agent_full.py
    tools=herramientas_totales,       # memoria + web + arxiv + instalar_skill + generar_grafico
    middleware=middlewares_activos,   # pipeline togglable actual (11 middlewares)
    subagents=[investigador],         # nuevo
    backend=FilesystemBackend(root_dir=RAIZ_PROYECTO),
    skills=["skills/"],              # rutas POSIX relativas al root del backend
    checkpointer=SqliteSaver(...),    # corto plazo, igual que hoy
    store=AlmacenPersistenteSQLite(...),  # largo plazo, igual que hoy
    interrupt_on={...},               # escrituras de archivo si humano_en_bucle activo
)
```

- Backend root = carpeta del proyecto. El agente puede `ls/read_file/write_file/edit_file` bajo esa raíz.
- `interrupt_on` para `write_file`/`edit_file` se liga al switch existente *Humano en el Bucle*.

### 2 · Paridad total con `agent_full.py`

Se conserva sin pérdida:
- Memoria dual: `SqliteSaver` (hilo) + `AlmacenPersistenteSQLite` (hechos durables, búsqueda híbrida semántica/keyword/recencia).
- Reflexión autónoma post-turno (`SalidaReflexion` con structured output, add/update/delete).
- Tools: `recordar`, `evocar`, `olvidar`, `buscar_en_red`, `investigar_a_fondo`, `extraer_pagina_web`, `search_arxiv`.
- `inyectar_memoria_dinamica` (`@dynamic_prompt`): recuerdos inyectados al system prompt. En DeepAgents se añade como middleware más el `system_prompt=` base del personaje.
- Panel de sliders LLM (temperatura, top-p, max tokens, razonamiento CoT).
- Panel de switches con los 11 middlewares togglables.
- Diagrama Mermaid del grafo, chat `mo.ui.chat`, inspector de memoria, visor de inyección de prompt, dashboard de estado, editor de herramientas del estudiante.

### 3 · Skills (nuevo)

- Directorio `./skills/` en el proyecto. Cada skill: subcarpeta con `SKILL.md` (frontmatter YAML `name` + `description` + instrucciones markdown), opcionalmente scripts/referencias/plantillas. Estándar agentskills.io — mismo formato que Claude Code.
- Carga vía `skills=["skills/"]`: el agente ve solo los frontmatter al inicio y lee el contenido completo bajo demanda (progressive disclosure).
- Tool nueva `instalar_skill(fuente, nombre)`:
  - `fuente` acepta: nombre corto de skill en repos conocidos (`anthropics/skills`, `langchain-ai/langchain-skills`), URL de carpeta GitHub, o URL raw de un `SKILL.md`.
  - Descarga a `./skills/<nombre>/`. Sin git; usa la API de contenidos de GitHub / descarga raw.
  - Devuelve string de éxito/error; nunca lanza excepción al agente.
- Una skill de ejemplo empaquetada (escrita por el propio notebook si `./skills/` está vacío) para demostrar el flujo.

### 4 · Panel UI de Skills (nuevo)

Celda Marimo con:
- Tabla de skills instaladas (nombre, descripción — parseadas del frontmatter; malformadas se listan con aviso y se omiten de la carga).
- Input de fuente + botón instalar.
- Botón recargar → invalida la celda del agente y lo reconstruye reactivamente.

### 5 · Subagentes

Los subagentes se definen mediante el constructor de la sección 9 y se pasan a `create_deep_agent(subagents=...)`. El agente principal delega vía la `task` tool sin contaminar su contexto. Como ejemplo sembrado (si `./subagentes/` está vacío) se crea el `investigador`: `tools=[buscar_en_red, investigar_a_fondo, extraer_pagina_web, search_arxiv]` con prompt propio de investigación.

### 6 · Manejo de errores

- Sin `NVIDIA_API_KEY` → agente inactivo con mensaje instructivo (patrón actual).
- Instalación de skill fallida → string de error legible.
- Skill malformada (sin frontmatter, sin `name`/`description`) → omitida de la carga, visible con aviso en el panel.
- Reflexión autónoma → errores silenciosos (patrón actual).

### 7 · Dependencias y verificación

- Header uv del script: añadir `deepagents` (y `pyyaml` si el parseo de frontmatter lo requiere; deepagents ya lo arrastra).
- Smoke test: `uv run python` que construya el agente (con y sin API key), parsee skills de ejemplo y verifique la tool `instalar_skill` contra un SKILL.md local.

### 8 · Salidas multimodales (nuevo)

**8a · Directorio de artefactos.** `./artefactos/` bajo el root del backend. Convención: toda tool (o el agente vía `write_file`) que produzca contenido no-textual lo guarda ahí y devuelve la ruta. Tool de ejemplo `generar_grafico(datos, tipo)` que produce PNG (Altair, ya en deps) para demostrar el flujo.

**8b · Galería de Artefactos (celda nueva).** Escanea `./artefactos/`, renderiza por extensión:

| Extensión | Render |
|---|---|
| png/jpg/webp/svg/gif | `mo.image` |
| pdf | `mo.pdf` (visor embebido) |
| mp4/webm | `mo.video` |
| mp3/wav | `mo.audio` |
| csv/parquet | `mo.ui.table` vía Polars |
| json/md/html | render nativo |
| otro | `mo.download` |

Botón refrescar, orden por fecha de modificación descendente, límite configurable de N recientes, cada artefacto en `mo.accordion` con nombre/tamaño/fecha.

**8c · Chat multimodal.** `ejecutar_agente` post-procesa la respuesta: detecta rutas a artefactos nuevas o mencionadas durante el turno y hace yield del objeto Marimo correspondiente inline en el chat, además del texto.

*Nota:* los modelos NIM configurados generan solo texto; lo multimodal proviene de tools. El panel queda preparado para modelos multimodales futuros.

### 9 · Constructor de Subagentes — "Reparto de la Obra" (nuevo)

**9a · Persistencia.** Directorio `./subagentes/`, un archivo `.md` por personaje, formato compatible con los agents de Claude Code (markdown + frontmatter YAML):

```markdown
---
name: investigador
description: Delega aquí investigación web profunda y búsqueda académica.
tools: [buscar_en_red, investigar_a_fondo, extraer_pagina_web, search_arxiv]
model: razonamiento   # opcional: estandar | razonamiento
---
Eres un investigador meticuloso. Contrastas fuentes, citas URLs...
```

Frontmatter = ficha técnica (el `description` es lo que el agente principal lee para decidir delegar); cuerpo = persona/system prompt del personaje.

**9b · Panel constructor (celda nueva).** Formulario Marimo:
- Nombre, descripción (rol en la obra), textarea de persona.
- Multiselect de tools disponibles (de `herramientas_totales`).
- Dropdown de modelo (estándar/razonamiento/heredar).
- Botones: guardar (escribe el `.md`), eliminar, dropdown para cargar/editar existente.
- Tabla del reparto actual (nombre + descripción + tools).

**9c · Integración.** Al arrancar y tras cada guardado se parsean todos los `./subagentes/*.md` → lista de dicts `SubAgent` → `create_deep_agent(subagents=...)`. Marimo reconstruye el agente reactivamente y el diagrama Mermaid refleja los subagentes nuevos. Archivos malformados → omitidos con aviso en el panel (mismo patrón que skills).

## Fuera de alcance

- Modificar `agent_full.py` o `cerebro_en_el_frasco.py`.
- Ejecución de scripts empaquetados en skills (el agente puede leerlos; ejecutarlos queda a discreción de tools existentes).
- Marketplace propio con índice/búsqueda; solo instalación directa desde fuentes conocidas o URL.
