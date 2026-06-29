# AGENTS.md — Agente01

## Project Overview
Python console agent using LangChain + NVIDIA NIM (Nemotron models) with persistent memory (filesystem-backed narrative memory + structured concepts + chat history). Uses Rich for rich terminal output.

## Quick Start
```powershell
uv sync                          # install deps
$env:NVIDIA_API_KEY="your_key"   # required (also TAVILY_API_KEY for web tools)
uv run python .\main.py          # run agent
```

## Commands
| Action | Command |
|--------|---------|
| Install deps | `uv sync` |
| Run agent | `uv run python .\main.py` |
| Syntax check | `.venv\Scripts\python.exe -c "import ast, sys; ast.parse(open('main.py').read()); print('OK')"` |
| Run tests | `uv run pytest` (no tests exist yet) |

## Key Files
| Path | Purpose |
|------|---------|
| `main.py` | Entry point: loads memory, initializes LLMs + tools, runs REPL |
| `pyproject.toml` | Project metadata + deps (uv-managed) |
| `memoria.md` | Persistent narrative memory (user facts, conclusions) |
| `memorias/conceptos.json` | Structured key-value concepts (created at runtime) |
| `memorias/historial.json` | Chat history (HumanMessage/AIMessage serialized to JSON) |
| `.env` | API keys (NVIDIA_API_KEY, TAVILY_API_KEY, etc.) |

## Architecture
- **Entry point**: `main.py` runs a REPL loop
- **LLMs**: Two `ChatNVIDIA` instances (`llm` = Nemotron-3-Ultra, `llm2` = Nemotron-3-Nano-Omni-Reasoning)
- **Tools**: Web search/crawl (Tavily), current time, persistent memory (read/write narrative, structured concepts)
- **Agent**: `create_tool_calling_agent` + `AgentExecutor` with chat history + system prompt in Spanish
- **Memory persistence**: JSON files under `memorias/` (auto-created on first run), `memoria.md` appended on writes
- **Output**: Rich panels with markdown rendering for reasoning + final answer

## Environment Variables (`.env`)
| Variable | Required | Purpose |
|----------|----------|---------|
| `NVIDIA_API_KEY` | Yes | NVIDIA NIM API access |
| `TAVILY_API_KEY` | Yes (for web tools) | Tavily search/crawl API |
| Others in `.env` | Optional | Other API keys (unused by default) |

## Conventions & Gotchas
- **Paths are hardcoded** to `D:\Documentos\CUGDL\Curso_Agentes\Agente01` in `main.py:22` — change if repo moves
- **`memorias/` directory auto-created** on first run (`main.py:26`)
- **No test suite exists** — add tests under `tests/` if needed
- **Python 3.13+ required** (`pyproject.toml:6`)
- **Dependencies managed by uv** (`uv.lock` committed)
- **System prompt in Spanish** — agent responds formally/concisely in Spanish
- **Two LLMs configured** but only `llm` used in `AgentExecutor` (line 190); `llm2` defined but unused
- **Reasoning enabled** via `chat_template_kwargs={"enable_thinking": True}` — Rich renders thinking blocks separately
- **Memory tools**: `leer_memoria`, `escribir_en_memoria`, `guardar_concepto` — agent must be prompted to use them
- **No tests, lint, typecheck, or CI configured** — add if needed

## Adding Tools
Define `@tool` functions in `main.py` (see `search`, `crawl`, `hora_actual`, `leer_memoria`, `escribir_en_memoria`, `guardar_concepto`). Add to `tools` list passed to `AgentExecutor`.

## Extending Memory
- Narrative: append to `memoria.md` via `escribir_en_memoria`
- Structured: key-value JSON via `guardar_concepto` / `conceptos.json`
- History: auto-saved per session to `historial.json`

## Known Gaps
- No test suite, lint, typecheck, or CI
- Hardcoded Windows path in `main.py:22`
- `llm2` defined but unused
- `TAVILY_API_KEY` required for web tools but not validated at startup
- No command-line args (model, memory path, etc. hardcoded)