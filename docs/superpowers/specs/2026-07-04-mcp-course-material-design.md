# Diseño: Integración MCP + Material Educativo Completo — agent_deep.py

**Fecha:** 2026-07-04
**Estado:** Aprobado por secciones en diálogo de brainstorming
**Alcance:** Añadir CONCEPTO 9 · MCP (cliente + servidor demo propio + panel + radiografía del protocolo) a `agent_deep.py`, y expandir el material educativo de los 9 conceptos (teoría narrativa + diagrama mermaid por concepto, inline y colapsable). `agent_full.py` y `cerebro_en_el_frasco.py` quedan intactos.

---

## 1. Contexto y objetivo

`agent_deep.py` (DeepAgents sobre Marimo) ya cubre 8 conceptos: agente, memoria dual, tools, middlewares, deep agents, skills, subagentes, multimodal. El usuario quiere:

- **A. MCP (Model Context Protocol)** integrado al agente con la misma estructura educativa: el agente consume servidores MCP Y el estudiante ve ambos lados del protocolo mediante un servidor demo propio legible.
- **B. Material de curso completo**: cada concepto (1-9) gana teoría narrativa (300-600 palabras, analogía central) + diagrama mermaid, inline en el notebook, colapsado por defecto.

Decisiones del usuario en diálogo:
- Alcance MCP: **cliente + servidor propio** (ambos lados del protocolo).
- Material educativo: **inline en notebook, colapsable** (el notebook ES el curso).
- Componentes por concepto: **teoría narrativa + analogía** y **diagrama mermaid** (sin ejercicios ni quiz).
- Gestión de servidores externos: **panel formulario Marimo + archivo `mcp_config.json`** (ambos).
- Enfoque técnico: **C — Híbrido**: adapters oficiales (`langchain-mcp-adapters`) para la integración real + una celda "radiografía" acotada que muestra los mensajes JSON-RPC crudos del protocolo.

## 2. Archivos

| Archivo | Rol |
|---|---|
| `servidor_mcp.py` (nuevo) | Servidor MCP demo con FastMCP (stdio), ~70 líneas legibles: 2-3 tools de laboratorio (`consultar_glosario` del curso, `estadisticas_curso` que lee ./skills y ./subagentes, `convertir_unidades`) + 1 resource (`curso://glosario`). Comentarios educativos en español. |
| `mcp_soporte.py` (nuevo) | Lógica pura (stdlib + json): leer/validar/escribir `mcp_config.json`, CRUD (`agregar_servidor`, `eliminar_servidor`, `listar_servidores`), validación de nombres (patrón `_nombre_seguro`), normalización al formato `MultiServerMCPClient`, normalización `"python"` → `sys.executable`. Sin marimo, sin langchain, sin red. |
| `mcp_config.json` (nuevo, sembrado) | Config declarativa estilo `claude_desktop_config.json`. Semilla: `{"mcpServers": {"laboratorio": {"transport": "stdio", "command": "python", "args": ["servidor_mcp.py"], "enabled": true}}}` — demo garantizada sin dependencias externas. Soporta `"enabled": false` para deshabilitar sin borrar. |
| `tests/test_mcp_soporte.py` (nuevo) | Tests pytest del módulo puro, sin red ni subprocess. |
| `agent_deep.py` (modificado) | CONCEPTO 9 (banner + módulo educativo + celdas cliente/panel/radiografía), módulos educativos conceptos 1-8, índice actualizado, migración chat a async. |
| `pyproject.toml` + header uv del notebook | + `langchain-mcp-adapters`, + `mcp` (trae FastMCP). |

## 3. Flujo de integración MCP

```
mcp_config.json ──lee──▶ mcp_soporte (validar/normalizar/filtrar enabled)
    ──▶ MultiServerMCPClient(config, tool_name_prefix=True)
    ──▶ await client.get_tools()   (por servidor, con try/except individual)
    ──▶ tools MCP se suman a herramientas_totales
    ──▶ create_deep_agent(...) las recibe igual que las nativas
Recargar (panel) ──▶ bump mo.state version_mcp ──▶ celda del agente se reconstruye
```

- `tool_name_prefix=True` evita colisiones de nombres entre servidores.
- Descubrimiento **por servidor** (cliente individual o try/except por subconjunto): un servidor caído aparece 🔴 con su error resumido; los demás siguen. Sin config o vacía → agente funciona como hoy.
- Timeout 15s por servidor al descubrir tools (primera ejecución de `npx`/`uvx` descarga paquetes; se documenta en el panel).

## 4. Panel MCP (Marimo)

Misma estética y patrones que Panel de Skills / Reparto:

1. **Tabla de servidores** — nombre, transporte, comando/URL, estado (🟢 N tools / 🔴 error resumido / ⚪ deshabilitado), tools descubiertas por servidor.
2. **Formulario añadir** — nombre, transporte (dropdown stdio/http/sse), comando+args o URL según transporte, env opcional (textarea `CLAVE=valor` por línea). Guardar → valida en `mcp_soporte` → persiste JSON → bump `mo.state`.
3. **Eliminar** — dropdown servidor + botón (patrón reparto).
4. **Recargar** — relee JSON, reconecta, reconstruye agente.
5. Callout permanente: primera conexión de servidores `npx`/`uvx` puede tardar (descarga).

## 5. Celda "Radiografía del protocolo"

Celda educativa autocontenida bajo CONCEPTO 9, con `mo.ui.run_button("🔬 Ejecutar radiografía")`:

1. Lanza `servidor_mcp.py` como subprocess stdio (SDK `mcp`: `stdio_client` + `ClientSession`), capturando los JSON crudos mediante interceptor sobre los streams.
2. Secuencia: `initialize` → `notifications/initialized` → `tools/list` → `tools/call` (tool demo, args fijos).
3. Render: tabla/accordion de 3 pasos con request y response JSON pretty-printed + anotación educativa por paso (negociación de capabilities / catálogo de tools / invocación).

Implementación primaria: interceptar streams del SDK. Fallback documentado si resulta frágil: JSON-RPC directo con `subprocess.Popen` (3 mensajes fijos). Ambas ≤60 líneas; decisión final en fase de plan tras prueba. Subprocess muere al terminar la celda (context manager). No toca el agente principal.

## 6. Material educativo (conceptos 1-9)

Patrón uniforme — una celda nueva por concepto tras su banner:

```python
mo.accordion({
    "📖 Teoría: <título>": mo.md(TEORIA_N),
    "🗺️ Diagrama": mo.mermaid(DIAGRAMA_N),
})
```

| # | Concepto | Analogía | Diagrama |
|---|----------|----------|----------|
| 1 | Agente de IA | Cerebro en el frasco | flowchart: ciclo percibir→razonar→actuar→observar (ReAct) |
| 2 | Memoria dual | Cuaderno vs biblioteca | flowchart: checkpointer/thread vs store/namespace, write-through SQLite |
| 3 | Tools | Manos del cerebro | sequence: LLM → tool_call → ejecutor → ToolMessage → LLM |
| 4 | Middlewares | Aduanas del pipeline | flowchart LR: petición atravesando capas |
| 5 | Deep Agents | Director de orquesta con partitura | flowchart: planning + filesystem + summarization sobre LangGraph |
| 6 | Skills | Manuales en la estantería | flowchart: progressive disclosure |
| 7 | Subagentes | Reparto de una obra | flowchart: director delega vía task() a actores con contexto limpio |
| 8 | Multimodal | Taller con mesa de resultados | flowchart: tool → artefacto → clasificador → galería/chat |
| 9 | MCP | Puerto USB-C de los agentes | sequence (initialize/tools-list/tools-call) + flowchart host/cliente/servidor |

Extras:
- Celda índice al inicio: tabla de los 9 conceptos ↔ celda que los demuestra (actualiza tabla existente `| Módulo | Concepto |`).
- Banners de comentarios existentes quedan como resumen rápido; teoría larga vive en el accordion.
- Español, tono educativo del curso, emojis moderados.
- Teoría de CONCEPTO 9 referencia las piezas: config, panel, radiografía, servidor demo, y explica la latencia de sesión-por-llamada.

## 7. Migración async

Tools MCP de los adapters son `StructuredTool(coroutine=...)` — solo async. El chat sync actual (`agente_cerebro.invoke`) lanzaría `NotImplementedError` al llamar una tool MCP.

1. `ejecutar_agente` → `async def` + `await agente_cerebro.ainvoke(...)`. Marimo soporta callbacks async en `mo.ui.chat` y top-level `await` en celdas — verificar en fase de plan; fallback: wrapper `asyncio.run()` en callback sync.
2. Celda de descubrimiento MCP: `await client.get_tools()` (top-level await).
3. Reflexión autónoma (`llm_estructurado.invoke`) no cambia — LLM directo sin tools.

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Cada tool-call MCP abre sesión stdio nueva → latencia | Aceptado (robusto, sin estado colgante); documentado en teoría 9 |
| `command: "python"` resuelve fuera del venv | `mcp_soporte` normaliza a `sys.executable` al cargar (pura, testeable) |
| Primera conexión `npx`/`uvx` descarga → cuelgue aparente | Timeout 15s + callout en panel |
| Windows ProactorEventLoop para subprocess async | Default moderno; smoke test en plan |
| `mcp`/adapters no instalados (headless) | Import guard: aviso en panel, agente sigue sin MCP |
| Nombres de servidor inseguros en formulario | Validación `_nombre_seguro` en `mcp_soporte` antes de persistir |

## 9. Testing y verificación

- `tests/test_mcp_soporte.py`: CRUD config; nombres inseguros (`../evil`) rechazados; normalización `python`→`sys.executable`; `enabled:false` filtrado; JSON corrupto → aviso, no excepción; config ausente → lista vacía. Sin red, sin subprocess.
- Integración opcional `@pytest.mark.integration`: lanza `servidor_mcp.py` real, verifica tools demo en `tools/list`; se salta si `mcp` no instalado.
- Runtime: `ast.parse agent_deep.py`, `uv run agent_deep.py` exit 0 con y sin `NVIDIA_API_KEY`, suite previa (27 tests) verde.

## 10. Fuera de alcance

- Modificar `agent_full.py` o `cerebro_en_el_frasco.py`.
- Ejercicios prácticos y quiz de autoevaluación (descartados por el usuario).
- Resources/prompts MCP en el cliente del agente (los adapters se centran en tools; el resource del servidor demo existe para que el estudiante lo lea en el código y la teoría lo explique — no se consume desde el agente ni la radiografía).
- OAuth/autenticación de servidores MCP remotos.
