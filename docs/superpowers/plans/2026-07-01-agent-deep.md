# agent_deep.py (DeepAgents + Skills + Subagentes + Multimodal) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nuevo notebook Marimo `agent_deep.py` construido sobre `deepagents.create_deep_agent`, con paridad total con `agent_full.py`, más skills de marketplace (SKILL.md), constructor de subagentes tipo "reparto de obra" y galería de salidas multimodales.

**Architecture:** La lógica pura y testeable (parseo de frontmatter, instalador de skills desde GitHub, carga/guardado de subagentes `.md`, clasificación de artefactos) vive en un módulo nuevo `deep_soporte.py` con tests pytest. El notebook `agent_deep.py` importa ese módulo y aporta solo UI Marimo + wiring del agente. `agent_full.py` NO se toca.

**Tech Stack:** Python 3.11+, uv (script header inline), marimo, deepagents (`create_deep_agent`, `FilesystemBackend`), langchain/langgraph (middlewares existentes), langchain-nvidia-ai-endpoints, PyYAML, Altair + Polars (gráficos), pytest.

**Spec:** `docs/superpowers/specs/2026-07-01-agent-deep-deepagents-design.md`

## Global Constraints

- Idioma del código: nombres y docstrings en español, estilo educativo con comentarios de concepto (igual que `agent_full.py`).
- NO modificar `agent_full.py` ni `cerebro_en_el_frasco.py`.
- `agent_deep.py` lleva header uv inline con TODAS las deps de `agent_full.py:1-31` más `deepagents` y `pyyaml`.
- `deep_soporte.py` solo usa stdlib + PyYAML (sin marimo, sin langchain) para que los tests corran rápido.
- NO añadir `TodoListMiddleware` al agente deep: DeepAgents ya incluye planning (`write_todos`) en su stack por defecto; duplicarlo rompe el registro de tools.
- El middleware de memoria dinámica debe APPENDEAR al `system_prompt` existente de la petición, nunca reemplazarlo (preservar el prompt de andamiaje de DeepAgents).
- Rutas de skills en `create_deep_agent(skills=[...])` son POSIX relativas al root del backend: usar `"skills/"`.
- Directorios de datos del proyecto: `./skills/`, `./subagentes/`, `./artefactos/` bajo la carpeta del notebook (`mo.notebook_dir()`).
- Tests: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v` (no depende del venv del proyecto).
- Commits frecuentes, mensajes en inglés convencional.

---

### Task 1: `deep_soporte.py` — frontmatter y skills locales

**Files:**
- Create: `deep_soporte.py`
- Test: `tests/test_deep_soporte.py`

**Interfaces:**
- Produces:
  - `parsear_frontmatter(texto: str) -> tuple[dict | None, str]` — (frontmatter, cuerpo); `None` si falta/ inválido/ no-dict.
  - `listar_skills(directorio: Path) -> tuple[list[dict], list[str]]` — lista de `{"name","description","ruta"}` + avisos de malformadas.
  - `sembrar_skill_ejemplo(directorio: Path) -> Path | None` — crea skill ejemplo si el dir está vacío/no existe; devuelve ruta creada o `None`.

- [ ] **Step 1: Escribir tests que fallan**

```python
# tests/test_deep_soporte.py
"""Tests del módulo de soporte de agent_deep.py."""

from pathlib import Path

import deep_soporte as ds


# ── parsear_frontmatter ──────────────────────────────────────────────────────

FM_VALIDO = """---
name: mi-skill
description: Hace algo útil.
---
Cuerpo de instrucciones.
"""


def test_frontmatter_valido():
    fm, cuerpo = ds.parsear_frontmatter(FM_VALIDO)
    assert fm == {"name": "mi-skill", "description": "Hace algo útil."}
    assert cuerpo.strip() == "Cuerpo de instrucciones."


def test_frontmatter_ausente():
    fm, cuerpo = ds.parsear_frontmatter("solo texto plano")
    assert fm is None
    assert cuerpo == "solo texto plano"


def test_frontmatter_yaml_invalido():
    fm, _ = ds.parsear_frontmatter("---\n{ :rotísimo\n---\ncuerpo")
    assert fm is None


def test_frontmatter_no_dict():
    fm, _ = ds.parsear_frontmatter("---\n- a\n- b\n---\ncuerpo")
    assert fm is None


# ── listar_skills ────────────────────────────────────────────────────────────


def _crear_skill(base: Path, nombre: str, contenido: str):
    d = base / nombre
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(contenido, encoding="utf-8")


def test_listar_skills_valida_y_malformada(tmp_path):
    _crear_skill(tmp_path, "buena", FM_VALIDO)
    _crear_skill(tmp_path, "mala", "sin frontmatter")
    (tmp_path / "sin_skill_md").mkdir()

    skills, avisos = ds.listar_skills(tmp_path)

    assert len(skills) == 1
    assert skills[0]["name"] == "mi-skill"
    assert skills[0]["description"] == "Hace algo útil."
    assert Path(skills[0]["ruta"]).name == "SKILL.md"
    assert len(avisos) == 1
    assert "mala" in avisos[0]


def test_listar_skills_directorio_inexistente(tmp_path):
    skills, avisos = ds.listar_skills(tmp_path / "no_existe")
    assert skills == [] and avisos == []


# ── sembrar_skill_ejemplo ────────────────────────────────────────────────────


def test_sembrar_skill_ejemplo_en_dir_vacio(tmp_path):
    ruta = ds.sembrar_skill_ejemplo(tmp_path / "skills")
    assert ruta is not None and ruta.exists()
    skills, avisos = ds.listar_skills(tmp_path / "skills")
    assert len(skills) == 1 and avisos == []


def test_sembrar_skill_ejemplo_no_pisa_existentes(tmp_path):
    base = tmp_path / "skills"
    _crear_skill(base, "existente", FM_VALIDO)
    assert ds.sembrar_skill_ejemplo(base) is None
```

- [ ] **Step 2: Correr tests — deben fallar**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: FAIL / ERROR con `ModuleNotFoundError: No module named 'deep_soporte'`.

- [ ] **Step 3: Implementación mínima**

```python
# deep_soporte.py
"""Funciones puras de soporte para agent_deep.py.

Separadas del notebook Marimo para poder testearlas con pytest sin arrancar
la UI ni las dependencias de LangChain. Solo stdlib + PyYAML.
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
from pathlib import Path

import yaml

# ═══════════════════════════════════════════════════════════════════════════
# Frontmatter (formato de SKILL.md y de subagentes .md — igual que Claude Code)
# ═══════════════════════════════════════════════════════════════════════════

_PATRON_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def parsear_frontmatter(texto: str) -> tuple[dict | None, str]:
    """Separa frontmatter YAML y cuerpo. Devuelve (None, texto) si es inválido."""
    m = _PATRON_FRONTMATTER.match(texto)
    if not m:
        return None, texto
    try:
        datos = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, texto
    if not isinstance(datos, dict):
        return None, texto
    return datos, m.group(2)


# ═══════════════════════════════════════════════════════════════════════════
# Skills locales (estándar agentskills.io: carpeta con SKILL.md)
# ═══════════════════════════════════════════════════════════════════════════


def listar_skills(directorio: Path | str) -> tuple[list[dict], list[str]]:
    """Escanea subcarpetas con SKILL.md. Devuelve (skills, avisos).

    Cada skill: {"name", "description", "ruta"}. Malformadas → aviso, omitidas.
    """
    directorio = Path(directorio)
    skills: list[dict] = []
    avisos: list[str] = []
    if not directorio.is_dir():
        return skills, avisos

    for carpeta in sorted(p for p in directorio.iterdir() if p.is_dir()):
        skill_md = carpeta / "SKILL.md"
        if not skill_md.is_file():
            continue
        fm, _ = parsear_frontmatter(skill_md.read_text(encoding="utf-8"))
        if not fm or "name" not in fm or "description" not in fm:
            avisos.append(
                f"⚠️ '{carpeta.name}': SKILL.md sin frontmatter válido "
                "(requiere name y description) — omitida"
            )
            continue
        skills.append(
            {
                "name": str(fm["name"]),
                "description": str(fm["description"]),
                "ruta": str(skill_md),
            }
        )
    return skills, avisos


_SKILL_EJEMPLO = """---
name: resumen-ejecutivo
description: Usa esta skill cuando el usuario pida resumir un documento, texto largo o investigación en formato de resumen ejecutivo profesional.
---

# Resumen Ejecutivo

## Instrucciones

1. Lee el material fuente completo antes de resumir.
2. Estructura SIEMPRE el resumen así:
   - **TL;DR** (2-3 frases).
   - **Hallazgos clave** (máx. 5 viñetas, cada una con dato concreto).
   - **Implicaciones** (qué significa para el lector).
   - **Acciones recomendadas** (numeradas, accionables).
3. Longitud máxima: 300 palabras. Sin relleno ni frases de cortesía.
4. Si el material tiene cifras, inclúyelas textualmente — no las redondees.
"""


def sembrar_skill_ejemplo(directorio: Path | str) -> Path | None:
    """Crea la skill de ejemplo si el directorio está vacío. Idempotente."""
    directorio = Path(directorio)
    if directorio.is_dir() and any(directorio.iterdir()):
        return None
    destino = directorio / "resumen-ejecutivo"
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / "SKILL.md"
    ruta.write_text(_SKILL_EJEMPLO, encoding="utf-8")
    return ruta
```

- [ ] **Step 4: Correr tests — deben pasar**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add deep_soporte.py tests/test_deep_soporte.py
git commit -m "feat: add deep_soporte module with frontmatter parsing and local skills"
```

---

### Task 2: `deep_soporte.py` — instalador de skills desde marketplace

**Files:**
- Modify: `deep_soporte.py` (añadir al final)
- Test: `tests/test_deep_soporte.py` (añadir al final)

**Interfaces:**
- Consumes: `parsear_frontmatter`, `listar_skills` (Task 1).
- Produces:
  - `instalar_skill_desde_fuente(fuente: str, nombre: str | None, dir_skills: Path, descargar=_descargar) -> str` — string de éxito `✅...` o fallo `❌...`; nunca lanza hacia el agente (el wrapper tool captura excepciones de red).
  - `REPOS_CONOCIDOS: list[tuple[str, str]]` — `[("anthropics", "skills"), ("langchain-ai", "langchain-skills")]`.
  - `_descargar(url: str) -> bytes` — inyectable en tests.

- [ ] **Step 1: Añadir tests que fallan**

```python
# tests/test_deep_soporte.py — AÑADIR AL FINAL
import json


def _fake_descargar_factory(mapa: dict[str, bytes]):
    """Devuelve un descargador falso que sirve respuestas desde un dict URL→bytes."""

    def _fake(url: str) -> bytes:
        if url in mapa:
            return mapa[url]
        raise OSError(f"404 simulado: {url}")

    return _fake


def test_instalar_desde_url_raw_skill_md(tmp_path):
    url = "https://raw.githubusercontent.com/x/y/main/skills/foo/SKILL.md"
    fake = _fake_descargar_factory({url: FM_VALIDO.encode()})

    msg = ds.instalar_skill_desde_fuente(url, None, tmp_path, descargar=fake)

    assert msg.startswith("✅")
    assert (tmp_path / "foo" / "SKILL.md").read_text(encoding="utf-8") == FM_VALIDO


def test_instalar_desde_carpeta_github(tmp_path):
    api = "https://api.github.com/repos/own/rep/contents/skills/bar?ref=main"
    listado = [
        {
            "type": "file",
            "name": "SKILL.md",
            "size": 100,
            "download_url": "https://raw.example/SKILL.md",
        },
        {"type": "dir", "name": "scripts", "path": "skills/bar/scripts"},
    ]
    api_sub = "https://api.github.com/repos/own/rep/contents/skills/bar/scripts?ref=main"
    sub_listado = [
        {
            "type": "file",
            "name": "run.py",
            "size": 50,
            "download_url": "https://raw.example/run.py",
        }
    ]
    fake = _fake_descargar_factory(
        {
            api: json.dumps(listado).encode(),
            api_sub: json.dumps(sub_listado).encode(),
            "https://raw.example/SKILL.md": FM_VALIDO.encode(),
            "https://raw.example/run.py": b"print('hola')",
        }
    )

    msg = ds.instalar_skill_desde_fuente(
        "https://github.com/own/rep/tree/main/skills/bar", None, tmp_path, descargar=fake
    )

    assert msg.startswith("✅")
    assert (tmp_path / "bar" / "SKILL.md").exists()
    assert (tmp_path / "bar" / "scripts" / "run.py").read_bytes() == b"print('hola')"


def test_instalar_nombre_corto_prueba_repos_conocidos(tmp_path):
    # 'anthropics/skills' falla en ambas rutas; 'langchain-ai/langchain-skills' acierta.
    api_ok = "https://api.github.com/repos/langchain-ai/langchain-skills/contents/baz?ref=main"
    listado = [
        {
            "type": "file",
            "name": "SKILL.md",
            "size": 10,
            "download_url": "https://raw.example/baz.md",
        }
    ]
    fake = _fake_descargar_factory(
        {api_ok: json.dumps(listado).encode(), "https://raw.example/baz.md": FM_VALIDO.encode()}
    )

    msg = ds.instalar_skill_desde_fuente("baz", None, tmp_path, descargar=fake)

    assert msg.startswith("✅")
    assert "langchain-ai/langchain-skills" in msg
    assert (tmp_path / "baz" / "SKILL.md").exists()


def test_instalar_nombre_corto_inexistente(tmp_path):
    fake = _fake_descargar_factory({})
    msg = ds.instalar_skill_desde_fuente("nadie", None, tmp_path, descargar=fake)
    assert msg.startswith("❌")
    assert not (tmp_path / "nadie").exists()


def test_instalar_carpeta_sin_skill_md_se_limpia(tmp_path):
    api = "https://api.github.com/repos/anthropics/skills/contents/qux?ref=main"
    listado = [
        {
            "type": "file",
            "name": "leeme.txt",
            "size": 5,
            "download_url": "https://raw.example/leeme.txt",
        }
    ]
    fake = _fake_descargar_factory(
        {api: json.dumps(listado).encode(), "https://raw.example/leeme.txt": b"hola"}
    )
    msg = ds.instalar_skill_desde_fuente("qux", None, tmp_path, descargar=fake)
    assert msg.startswith("❌")
    assert not (tmp_path / "qux").exists()
```

- [ ] **Step 2: Correr tests — los nuevos deben fallar**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: los 8 de Task 1 PASS; los 5 nuevos FAIL con `AttributeError: ... 'instalar_skill_desde_fuente'`.

- [ ] **Step 3: Implementar el instalador**

```python
# deep_soporte.py — AÑADIR AL FINAL

# ═══════════════════════════════════════════════════════════════════════════
# Instalador de skills desde el marketplace (repos GitHub de skills)
#
# Fuentes aceptadas:
#   1. URL raw a un SKILL.md            → skill de un solo archivo.
#   2. URL github.com/.../tree/<ref>/<ruta> → descarga la carpeta completa.
#   3. Nombre corto ('pdf', 'skills/pdf')   → se sondea en REPOS_CONOCIDOS.
# ═══════════════════════════════════════════════════════════════════════════

REPOS_CONOCIDOS: list[tuple[str, str]] = [
    ("anthropics", "skills"),
    ("langchain-ai", "langchain-skills"),
]
MAX_ARCHIVOS = 30        # tope de seguridad por instalación
MAX_BYTES = 1_000_000    # se omiten archivos > 1 MB (binarios pesados)

_PATRON_TREE_GITHUB = re.compile(
    r"https?://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)"
)


def _descargar(url: str) -> bytes:
    """Descarga una URL. Inyectable en tests para no tocar la red."""
    peticion = urllib.request.Request(
        url, headers={"User-Agent": "agent-deep-educativo"}
    )
    with urllib.request.urlopen(peticion, timeout=30) as respuesta:
        return respuesta.read()


def _descargar_carpeta_github(
    owner: str,
    repo: str,
    ruta: str,
    ref: str,
    destino: Path,
    descargar,
    _contador: list[int] | None = None,
) -> int:
    """Descarga recursiva de una carpeta vía la API de contenidos de GitHub."""
    if _contador is None:
        _contador = [0]
    url_api = f"https://api.github.com/repos/{owner}/{repo}/contents/{ruta}?ref={ref}"
    listado = json.loads(descargar(url_api))
    if isinstance(listado, dict):  # la ruta era un archivo suelto
        listado = [listado]

    destino.mkdir(parents=True, exist_ok=True)
    for entrada in listado:
        if _contador[0] >= MAX_ARCHIVOS:
            break
        if entrada["type"] == "dir":
            _descargar_carpeta_github(
                owner, repo, entrada["path"], ref,
                destino / entrada["name"], descargar, _contador,
            )
        elif entrada["type"] == "file":
            if entrada.get("size", 0) > MAX_BYTES:
                continue
            (destino / entrada["name"]).write_bytes(
                descargar(entrada["download_url"])
            )
            _contador[0] += 1
    return _contador[0]


def instalar_skill_desde_fuente(
    fuente: str,
    nombre: str | None,
    dir_skills: Path | str,
    descargar=_descargar,
) -> str:
    """Instala una skill en dir_skills. Devuelve mensaje ✅/❌ (no lanza al agente)."""
    dir_skills = Path(dir_skills)
    dir_skills.mkdir(parents=True, exist_ok=True)
    fuente = fuente.strip()

    # Caso 1 · URL directa a un SKILL.md
    if fuente.startswith("http") and fuente.rstrip("/").endswith("SKILL.md"):
        nombre_final = nombre or fuente.rstrip("/").split("/")[-2]
        destino = dir_skills / nombre_final
        destino.mkdir(parents=True, exist_ok=True)
        (destino / "SKILL.md").write_bytes(descargar(fuente))
        return f"✅ Skill '{nombre_final}' instalada (1 archivo) en {destino}"

    # Caso 2 · URL de carpeta github.com/.../tree/<ref>/<ruta>
    m = _PATRON_TREE_GITHUB.match(fuente)
    if m:
        owner, repo, ref, ruta = m.groups()
        nombre_final = nombre or ruta.rstrip("/").split("/")[-1]
        n = _descargar_carpeta_github(
            owner, repo, ruta, ref, dir_skills / nombre_final, descargar
        )
        return f"✅ Skill '{nombre_final}' instalada ({n} archivos)"

    # Caso 3 · Nombre corto → sondear repos conocidos del marketplace
    if not fuente.startswith("http"):
        nombre_final = nombre or fuente.split("/")[-1]
        destino = dir_skills / nombre_final
        for owner, repo in REPOS_CONOCIDOS:
            for ruta in (fuente, f"skills/{fuente}"):
                try:
                    n = _descargar_carpeta_github(
                        owner, repo, ruta, "main", destino, descargar
                    )
                except Exception:
                    continue
                if n and (destino / "SKILL.md").is_file():
                    return (
                        f"✅ Skill '{nombre_final}' instalada desde "
                        f"{owner}/{repo} ({n} archivos)"
                    )
                # Carpeta sin SKILL.md válido → limpiar y seguir sondeando
                shutil.rmtree(destino, ignore_errors=True)
        return (
            f"❌ No encontré la skill '{fuente}' en los repos conocidos: "
            + ", ".join(f"{o}/{r}" for o, r in REPOS_CONOCIDOS)
        )

    return (
        "❌ Fuente no reconocida. Usa: nombre corto, "
        "URL github.com/.../tree/<rama>/<ruta>, o URL raw a un SKILL.md."
    )
```

- [ ] **Step 4: Correr tests — todos pasan**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add deep_soporte.py tests/test_deep_soporte.py
git commit -m "feat: add marketplace skill installer to deep_soporte"
```

---

### Task 3: `deep_soporte.py` — subagentes ("reparto de la obra")

**Files:**
- Modify: `deep_soporte.py` (añadir al final)
- Test: `tests/test_deep_soporte.py` (añadir al final)

**Interfaces:**
- Consumes: `parsear_frontmatter` (Task 1).
- Produces:
  - `cargar_subagentes(directorio: Path, registro_tools: dict[str, object]) -> tuple[list[dict], list[str]]` — dicts con claves `name`, `description`, `system_prompt`, `tools` (objetos resueltos) y opcional `model_alias` (`"estandar"`/`"razonamiento"`); el notebook resuelve el alias a un LLM y lo renombra a `model`.
  - `guardar_subagente_md(directorio, name, description, persona, tools=(), model=None) -> Path`
  - `eliminar_subagente_md(directorio: Path, name: str) -> bool`
  - `sembrar_subagente_ejemplo(directorio: Path) -> Path | None` — crea `investigador.md` si el dir está vacío.

- [ ] **Step 1: Añadir tests que fallan**

```python
# tests/test_deep_soporte.py — AÑADIR AL FINAL

SUBAGENTE_MD = """---
name: critico
description: Delega aquí revisiones de calidad de textos.
tools: [tool_a, tool_inexistente]
model: razonamiento
---
Eres un crítico literario implacable pero justo.
"""


def test_cargar_subagentes(tmp_path):
    (tmp_path / "critico.md").write_text(SUBAGENTE_MD, encoding="utf-8")
    (tmp_path / "roto.md").write_text("sin frontmatter", encoding="utf-8")
    registro = {"tool_a": "OBJETO_TOOL_A"}

    subs, avisos = ds.cargar_subagentes(tmp_path, registro)

    assert len(subs) == 1
    sub = subs[0]
    assert sub["name"] == "critico"
    assert sub["description"].startswith("Delega")
    assert sub["system_prompt"].startswith("Eres un crítico")
    assert sub["tools"] == ["OBJETO_TOOL_A"]
    assert sub["model_alias"] == "razonamiento"
    # dos avisos: archivo roto + tool desconocida
    assert len(avisos) == 2


def test_cargar_subagentes_cuerpo_vacio_se_omite(tmp_path):
    (tmp_path / "vacio.md").write_text(
        "---\nname: vacio\ndescription: x\n---\n", encoding="utf-8"
    )
    subs, avisos = ds.cargar_subagentes(tmp_path, {})
    assert subs == [] and len(avisos) == 1


def test_guardar_y_recargar_subagente(tmp_path):
    ruta = ds.guardar_subagente_md(
        tmp_path,
        name="poeta",
        description="Delega aquí la escritura de poemas.",
        persona="Eres un poeta del Siglo de Oro.",
        tools=["tool_a"],
        model="estandar",
    )
    assert ruta.name == "poeta.md"
    subs, avisos = ds.cargar_subagentes(tmp_path, {"tool_a": 1})
    assert avisos == []
    assert subs[0]["name"] == "poeta"
    assert subs[0]["tools"] == [1]
    assert subs[0]["model_alias"] == "estandar"


def test_eliminar_subagente(tmp_path):
    ds.guardar_subagente_md(tmp_path, "x", "d", "persona", [], None)
    assert ds.eliminar_subagente_md(tmp_path, "x") is True
    assert ds.eliminar_subagente_md(tmp_path, "x") is False


def test_sembrar_subagente_ejemplo(tmp_path):
    ruta = ds.sembrar_subagente_ejemplo(tmp_path / "subs")
    assert ruta is not None and ruta.name == "investigador.md"
    # Idempotente: segunda siembra no hace nada
    assert ds.sembrar_subagente_ejemplo(tmp_path / "subs") is None
```

- [ ] **Step 2: Correr tests — los nuevos fallan**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: 13 previos PASS, 5 nuevos FAIL (`AttributeError: cargar_subagentes`).

- [ ] **Step 3: Implementar subagentes**

```python
# deep_soporte.py — AÑADIR AL FINAL

# ═══════════════════════════════════════════════════════════════════════════
# Subagentes — "El Reparto de la Obra"
#
# Cada personaje es un archivo .md con frontmatter YAML (mismo formato que
# los agents de Claude Code): name, description, tools (opcional),
# model (opcional: 'estandar' | 'razonamiento'). El cuerpo es la persona.
# ═══════════════════════════════════════════════════════════════════════════


def cargar_subagentes(
    directorio: Path | str, registro_tools: dict[str, object]
) -> tuple[list[dict], list[str]]:
    """Parsea ./subagentes/*.md → dicts listos para create_deep_agent(subagents=…).

    Los nombres de tools se resuelven contra registro_tools; desconocidas se
    ignoran con aviso. El alias de modelo se devuelve como 'model_alias' para
    que el notebook lo resuelva al objeto LLM correspondiente.
    """
    directorio = Path(directorio)
    subagentes: list[dict] = []
    avisos: list[str] = []
    if not directorio.is_dir():
        return subagentes, avisos

    for archivo in sorted(directorio.glob("*.md")):
        fm, cuerpo = parsear_frontmatter(archivo.read_text(encoding="utf-8"))
        if not fm or "name" not in fm or "description" not in fm:
            avisos.append(
                f"⚠️ '{archivo.name}': frontmatter inválido "
                "(requiere name y description) — omitido"
            )
            continue
        if not cuerpo.strip():
            avisos.append(f"⚠️ '{archivo.name}': sin persona (cuerpo vacío) — omitido")
            continue

        tools_resueltas: list[object] = []
        desconocidas: list[str] = []
        for nombre_tool in fm.get("tools") or []:
            if nombre_tool in registro_tools:
                tools_resueltas.append(registro_tools[nombre_tool])
            else:
                desconocidas.append(nombre_tool)
        if desconocidas:
            avisos.append(
                f"⚠️ '{archivo.name}': tools desconocidas {desconocidas} — ignoradas"
            )

        subagente: dict = {
            "name": str(fm["name"]),
            "description": str(fm["description"]),
            "system_prompt": cuerpo.strip(),
            "tools": tools_resueltas,
        }
        if fm.get("model"):
            subagente["model_alias"] = str(fm["model"])
        subagentes.append(subagente)
    return subagentes, avisos


def guardar_subagente_md(
    directorio: Path | str,
    name: str,
    description: str,
    persona: str,
    tools: list[str] | tuple = (),
    model: str | None = None,
) -> Path:
    """Escribe <directorio>/<name>.md con frontmatter + persona."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    frontmatter: dict = {"name": name, "description": description}
    if tools:
        frontmatter["tools"] = list(tools)
    if model:
        frontmatter["model"] = model
    contenido = (
        "---\n"
        + yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False)
        + "---\n"
        + persona.strip()
        + "\n"
    )
    ruta = directorio / f"{name}.md"
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


def eliminar_subagente_md(directorio: Path | str, name: str) -> bool:
    """Borra el archivo del personaje. True si existía."""
    ruta = Path(directorio) / f"{name}.md"
    if ruta.is_file():
        ruta.unlink()
        return True
    return False


_PERSONA_INVESTIGADOR = """Eres un investigador meticuloso y escéptico.

- Contrasta al menos dos fuentes antes de afirmar un hecho.
- Cita siempre las URLs de tus fuentes.
- Distingue explícitamente entre hecho verificado, consenso y especulación.
- Devuelve un informe estructurado: hallazgos, fuentes, grado de confianza.
"""


def sembrar_subagente_ejemplo(directorio: Path | str) -> Path | None:
    """Crea el personaje 'investigador' si el directorio está vacío."""
    directorio = Path(directorio)
    if directorio.is_dir() and any(directorio.glob("*.md")):
        return None
    return guardar_subagente_md(
        directorio,
        name="investigador",
        description=(
            "Delega aquí investigación web profunda y búsqueda académica: "
            "temas actuales, papers, verificación de hechos multi-fuente."
        ),
        persona=_PERSONA_INVESTIGADOR,
        tools=["buscar_en_red", "investigar_a_fondo", "extraer_pagina_web", "search_arxiv"],
        model="razonamiento",
    )
```

- [ ] **Step 4: Correr tests — todos pasan**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add deep_soporte.py tests/test_deep_soporte.py
git commit -m "feat: add subagent cast persistence to deep_soporte"
```

---

### Task 4: `deep_soporte.py` — clasificación de artefactos multimodales

**Files:**
- Modify: `deep_soporte.py` (añadir al final)
- Test: `tests/test_deep_soporte.py` (añadir al final)

**Interfaces:**
- Produces:
  - `clasificar_artefacto(ruta: Path) -> str` — una de: `"imagen" | "pdf" | "video" | "audio" | "tabla" | "json" | "texto" | "html" | "otro"`.
  - `listar_artefactos(directorio: Path, limite: int = 20) -> list[Path]` — archivos ordenados por mtime descendente.

- [ ] **Step 1: Añadir tests que fallan**

```python
# tests/test_deep_soporte.py — AÑADIR AL FINAL
import os
import time


def test_clasificar_artefacto():
    casos = {
        "a.png": "imagen", "b.JPG": "imagen", "c.svg": "imagen",
        "d.pdf": "pdf", "e.mp4": "video", "f.webm": "video",
        "g.mp3": "audio", "h.wav": "audio", "i.csv": "tabla",
        "j.parquet": "tabla", "k.json": "json", "l.md": "texto",
        "m.html": "html", "n.xyz": "otro",
    }
    for nombre, esperado in casos.items():
        assert ds.clasificar_artefacto(Path(nombre)) == esperado, nombre


def test_listar_artefactos_orden_y_limite(tmp_path):
    viejo = tmp_path / "viejo.png"
    nuevo = tmp_path / "nuevo.pdf"
    viejo.write_bytes(b"1")
    nuevo.write_bytes(b"2")
    ahora = time.time()
    os.utime(viejo, (ahora - 100, ahora - 100))
    os.utime(nuevo, (ahora, ahora))
    (tmp_path / "subdir").mkdir()  # los directorios se ignoran

    artefactos = ds.listar_artefactos(tmp_path)
    assert [p.name for p in artefactos] == ["nuevo.pdf", "viejo.png"]

    assert len(ds.listar_artefactos(tmp_path, limite=1)) == 1


def test_listar_artefactos_dir_inexistente(tmp_path):
    assert ds.listar_artefactos(tmp_path / "nada") == []
```

- [ ] **Step 2: Correr — los nuevos fallan**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: 18 PASS previos, 3 nuevos FAIL.

- [ ] **Step 3: Implementar**

```python
# deep_soporte.py — AÑADIR AL FINAL

# ═══════════════════════════════════════════════════════════════════════════
# Artefactos multimodales — clasificación por extensión para la galería
# ═══════════════════════════════════════════════════════════════════════════

_CATEGORIAS_ARTEFACTO: dict[str, set[str]] = {
    "imagen": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"},
    "pdf": {".pdf"},
    "video": {".mp4", ".webm"},
    "audio": {".mp3", ".wav"},
    "tabla": {".csv", ".parquet"},
    "json": {".json"},
    "texto": {".md", ".txt"},
    "html": {".html"},
}


def clasificar_artefacto(ruta: Path | str) -> str:
    """Categoría de render para la galería según la extensión del archivo."""
    extension = Path(ruta).suffix.lower()
    for categoria, extensiones in _CATEGORIAS_ARTEFACTO.items():
        if extension in extensiones:
            return categoria
    return "otro"


def listar_artefactos(directorio: Path | str, limite: int = 20) -> list[Path]:
    """Archivos del directorio (recursivo) ordenados del más reciente al más viejo."""
    directorio = Path(directorio)
    if not directorio.is_dir():
        return []
    archivos = [p for p in directorio.rglob("*") if p.is_file()]
    archivos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return archivos[:limite]
```

- [ ] **Step 4: Correr tests — todos pasan**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add deep_soporte.py tests/test_deep_soporte.py
git commit -m "feat: add multimodal artifact helpers to deep_soporte"
```

---

### Task 5: `agent_deep.py` — esqueleto del notebook, config y paridad de infraestructura

Crea el notebook con: header uv, celdas didácticas, configuración, paneles de control, almacén de memoria, checkpointer y LLMs. Los bloques idénticos a `agent_full.py` se copian VERBATIM de las líneas indicadas (el archivo fuente está en el repo; léelo antes de copiar).

**Files:**
- Create: `agent_deep.py`
- Reference (solo lectura): `agent_full.py`

**Interfaces:**
- Produces (nombres de celda/variables que Tasks 6-7 consumen): `mo`, `ds` (módulo deep_soporte), `RAIZ_PROYECTO`, `DIR_SKILLS`, `DIR_SUBAGENTES`, `DIR_ARTEFACTOS`, `ID_USUARIO`, `ESPACIO_MEMORIA`, `almacen_memoria`, `semantica_activa`, `gestor_puntos_control`, `llm_principal`, `llm_respaldo`, `llm_estandar_obj`, `nombre_modelo_activo`, `ui_*` (sliders), `menu_middlewares`, `mezclar_recuerdos`, `ultimo_texto_usuario`.

- [ ] **Step 1: Crear `agent_deep.py` con header y celdas base**

Header uv: copiar `agent_full.py:1-31` y añadir dos deps dentro de la lista:

```python
#     "deepagents>=0.2",
#     "pyyaml>=6",
```

Cabecera de comentarios didácticos: adaptar el banner de `agent_full.py:33-124` — mismo formato, título `AGENTE PROFUNDO · DeepAgents + Skills + Subagentes + Multimodal`, y añadir estos conceptos nuevos al final del bloque:

```python
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
```

Celdas (en orden; `import marimo` + `app = marimo.App(width="full")` como `agent_full.py:126-129`):

1. Celda imports Marimo: igual a `agent_full.py:132-136`.
2. Celda markdown de portada: adaptar `agent_full.py:139-164` (título `# 🧠 Agente Profundo — DeepAgents`, tabla de módulos añadiendo filas *Skills*, *Subagentes*, *Multimodal*).
3. Celda de imports pesados: copiar `agent_full.py:167-248` COMPLETA y añadir antes del `load_dotenv()`:

```python
    # ── DeepAgents ────────────────────────────────────────────────────────────────────
    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend

    # ── Módulo de soporte local (funciones puras testeadas con pytest) ───────────────
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import deep_soporte as ds
```

   y añadir `create_deep_agent`, `FilesystemBackend`, `ds`, `sys` al `return` de la celda. QUITAR del import y del return: `TodoListMiddleware` (ver Global Constraints) y `create_agent` (ya no se usa).
4. Celda de configuración: copiar `agent_full.py:252-289` y añadir al final (antes del `return`):

```python
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
```

   (añadir `ds` a los parámetros de la celda y las 4 rutas nuevas al return).
5. Paneles de control LLM: copiar `agent_full.py:292-345` verbatim.
6. Menú de middlewares: copiar `agent_full.py:348-398` QUITANDO la entrada `"planificacion_tareas"` y añadiendo en su lugar:

```python
            "filesystem_protegido": mo.ui.switch(
                value=False,
                label="📁 **Filesystem Protegido** — Pide aprobación antes de write_file / edit_file",
            ),
```

7. Celda hstack de paneles: copiar `agent_full.py:401-435` verbatim.
8. `AlmacenPersistenteSQLite`: copiar `agent_full.py:438-595` verbatim.
9. Celda de embeddings/almacén: copiar `agent_full.py:598-633` verbatim.
10. Celda checkpointer: copiar `agent_full.py:636-641` verbatim.
11. Celda LLMs: copiar `agent_full.py:644-680` con UNA adición — exponer también el LLM estándar como objeto reutilizable para subagentes:

```python
    llm_estandar_obj = llm_respaldo  # alias claro para el resolvedor de subagentes
```

   (añadir `llm_estandar_obj` al return).
12. Celdas de recuperación de memoria (`ultimo_texto_usuario`, `mezclar_recuerdos`): copiar `agent_full.py:683-720` verbatim.

- [ ] **Step 2: Verificar sintaxis**

Run: `uv run --no-project python -c "import ast; ast.parse(open('agent_deep.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agent_deep.py
git commit -m "feat: scaffold agent_deep notebook with config and memory parity"
```

---

### Task 6: `agent_deep.py` — tools, memoria dinámica, subagentes y construcción del agente

**Files:**
- Modify: `agent_deep.py` (añadir celdas tras las de Task 5)
- Reference: `agent_full.py`

**Interfaces:**
- Consumes: todo lo de Task 5.
- Produces: `herramientas_totales` (lista), `registro_tools` (dict nombre→tool), `instalar_skill`, `generar_grafico`, `inyectar_memoria_dinamica`, `PERSONAJE_BASE`, `middlewares_activos`, `middlewares_nombres`, `subagentes_cargados`, `avisos_subagentes`, `contador_reparto` (mo.state para recarga reactiva), `agente_cerebro`, `ID_HILO`, `arquitectura_mermaid`.

- [ ] **Step 1: Celda de tools de memoria**

Copiar `agent_full.py:723-809` verbatim (`recordar`, `evocar`, `olvidar`). NOTA: en `agent_full.py` la función `recordar` perdió su decorador `@tool` (bug preexistente — se pasa como función plana). En `agent_deep.py` corregirlo: decorar `recordar` con `@tool`.

- [ ] **Step 2: Celda de tools web + arXiv + nuevas tools**

Basarse en `agent_full.py:812-907` con estos cambios:
1. Añadir `@tool` a `buscar_en_red` (mismo bug preexistente; corregir).
2. Los imports `urllib`/`ET` van locales a esta celda (copiar los de `agent_full.py:1621-1623` al inicio de la celda) — así se elimina la celda duplicada del final del archivo original.
3. Añadir las dos tools nuevas ANTES de `herramientas_totales`:

```python
    @tool
    def instalar_skill(fuente: str, nombre: str = "") -> str:
        """Instala una skill del marketplace de Agent Skills en ./skills/.
        `fuente` puede ser: un nombre corto (se busca en anthropics/skills y
        langchain-ai/langchain-skills), una URL github.com/.../tree/<rama>/<ruta>,
        o una URL raw a un SKILL.md. `nombre` renombra la carpeta destino (opcional).
        Tras instalar, avisa al usuario que pulse 'Recargar' en el Panel de Skills."""
        try:
            return ds.instalar_skill_desde_fuente(fuente, nombre or None, DIR_SKILLS)
        except Exception as e:
            return f"❌ Error instalando skill: {e}"

    @tool
    def generar_grafico(
        datos_json: str,
        tipo: str = "barras",
        eje_x: str = "x",
        eje_y: str = "y",
        titulo: str = "Gráfico",
    ) -> str:
        """Genera un gráfico PNG en ./artefactos/ a partir de datos tabulares.
        `datos_json`: lista JSON de objetos, ej. '[{"x": "a", "y": 3}, ...]'.
        `tipo`: 'barras' | 'lineas' | 'puntos'. `eje_x`/`eje_y`: columnas a usar.
        Devuelve la ruta del PNG generado (aparece en la Galería y en el chat)."""
        import altair as alt
        import polars as pl

        try:
            df = pl.DataFrame(json.loads(datos_json))
            marcas = {"barras": "mark_bar", "lineas": "mark_line", "puntos": "mark_point"}
            metodo_marca = marcas.get(tipo, "mark_bar")
            grafico = getattr(alt.Chart(df, title=titulo), metodo_marca)().encode(
                x=eje_x, y=eje_y
            )
            nombre_archivo = (
                f"grafico_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
            )
            ruta = DIR_ARTEFACTOS / nombre_archivo
            grafico.save(str(ruta))  # requiere vl-convert (ya en deps)
            return f"📊 Gráfico guardado en artefactos/{nombre_archivo}"
        except Exception as e:
            return f"❌ Error generando gráfico: {e}"
```

4. Cerrar la celda con el registro (fuente única de verdad para subagentes y agente):

```python
    herramientas_totales = [
        recordar,
        evocar,
        olvidar,
        buscar_en_red,
        investigar_a_fondo,
        extraer_pagina_web,
        search_arxiv,
        instalar_skill,
        generar_grafico,
    ]
    # Registro nombre→tool: lo usan el constructor de subagentes y su cargador
    registro_tools = {t.name: t for t in herramientas_totales}
    return herramientas_totales, registro_tools, instalar_skill, generar_grafico
```

- [ ] **Step 3: Celda de memoria dinámica (modo APPEND)**

Basada en `agent_full.py:911-941` pero SIN reemplazar el system prompt de DeepAgents:

```python
@app.cell
def _(ID_USUARIO, dynamic_prompt, mezclar_recuerdos, ultimo_texto_usuario):
    PERSONAJE_BASE = (
        "Eres el Agente Profundo: un asistente lúcido, analítico, con memoria "
        "persistente, skills instalables y un reparto de subagentes a tu cargo. "
        "Hablas en el idioma del usuario, tono directo pero cálido. "
        "Cuando aprendas algo nuevo y duradero sobre el usuario, llama a `recordar`. "
        "Delega en tus subagentes (tool `task`) cuando su description encaje con la tarea."
    )

    @dynamic_prompt
    def inyectar_memoria_dinamica(peticion) -> str:
        """APPENDEA recuerdos al system prompt existente.

        ⚠️ Clave con DeepAgents: el prompt entrante ya contiene el andamiaje
        (planning, filesystem, task, skills). Reemplazarlo lo rompería;
        por eso concatenamos en vez de sustituir.
        """
        prompt_existente = peticion.system_prompt or ""
        consulta = ultimo_texto_usuario(peticion.messages) or ID_USUARIO
        recuerdos = mezclar_recuerdos(consulta)

        if recuerdos:
            bloque = "\n".join(f"  - {t}" for t in recuerdos)
            seccion = (
                f"\n\n## Lo que recuerdas de {ID_USUARIO} (memoria persistente):\n"
                f"{bloque}\n\n"
                "Usa estos recuerdos con naturalidad, sin anunciarlos. "
                "Si detectas información desactualizada, usa `olvidar` + `recordar`."
            )
        else:
            seccion = (
                f"\n\nAún no tienes recuerdos de {ID_USUARIO}. "
                "Cuando aprendas algo duradero, llama a `recordar`."
            )
        return prompt_existente + seccion

    return PERSONAJE_BASE, inyectar_memoria_dinamica
```

- [ ] **Step 4: Celda de middlewares**

Copiar `agent_full.py:944-1068` con estos cambios:
- Quitar el bloque `if opciones["planificacion_tareas"]:` (y el import ya se quitó en Task 5).
- Quitar `TodoListMiddleware` de los parámetros de la celda.
- El resto de switches/middlewares idéntico (incluido `HumanInTheLoopMiddleware` para tools de memoria/web).

- [ ] **Step 5: Celda de carga de subagentes**

```python
@app.cell
def _(DIR_SUBAGENTES, ds, llm_estandar_obj, llm_principal, mo, registro_tools):
    # Estado reactivo: incrementarlo desde el panel constructor fuerza recarga
    obtener_version_reparto, marcar_version_reparto = mo.state(0)

    def cargar_reparto():
        """Lee ./subagentes/*.md y resuelve alias de modelo a objetos LLM."""
        crudos, avisos = ds.cargar_subagentes(DIR_SUBAGENTES, registro_tools)
        subagentes = []
        for sub in crudos:
            sub = dict(sub)
            alias = sub.pop("model_alias", None)
            if alias == "estandar" and llm_estandar_obj is not None:
                sub["model"] = llm_estandar_obj
            elif alias == "razonamiento" and llm_principal is not None:
                sub["model"] = llm_principal
            # sin alias → hereda el modelo del agente principal
            subagentes.append(sub)
        return subagentes, avisos

    return cargar_reparto, marcar_version_reparto, obtener_version_reparto
```

- [ ] **Step 6: Celda de construcción del agente**

Reemplaza el patrón de `agent_full.py:1071-1092`:

```python
@app.cell
def _(
    FilesystemBackend,
    RAIZ_PROYECTO,
    almacen_memoria,
    cargar_reparto,
    create_deep_agent,
    gestor_puntos_control,
    herramientas_totales,
    llm_principal,
    menu_middlewares,
    middlewares_activos,
    obtener_version_reparto,
    PERSONAJE_BASE,
    uuid,
):
    _ = obtener_version_reparto()  # dependencia reactiva: recarga al guardar personajes
    subagentes_cargados, avisos_subagentes = cargar_reparto()

    agente_cerebro = None
    if llm_principal is not None:
        _interrupciones_fs = (
            {"write_file": True, "edit_file": True}
            if menu_middlewares.value["filesystem_protegido"]
            else None
        )
        agente_cerebro = create_deep_agent(
            model=llm_principal,
            system_prompt=PERSONAJE_BASE,
            tools=herramientas_totales,
            middleware=middlewares_activos,
            subagents=subagentes_cargados,
            backend=FilesystemBackend(root_dir=str(RAIZ_PROYECTO)),
            skills=["skills/"],
            interrupt_on=_interrupciones_fs,
            checkpointer=gestor_puntos_control,
            store=almacen_memoria,
        )

    ID_HILO = "sesion-" + uuid.uuid4().hex[:8]
    return ID_HILO, agente_cerebro, avisos_subagentes, subagentes_cargados
```

- [ ] **Step 7: Celdas de diagrama Mermaid**

Copiar `agent_full.py:1095-1134` verbatim (funcionan igual: el agente deep es un grafo LangGraph compilado).

- [ ] **Step 8: Verificar sintaxis y commit**

Run: `uv run --no-project python -c "import ast; ast.parse(open('agent_deep.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

```bash
git add agent_deep.py
git commit -m "feat: wire create_deep_agent with tools, skills, and subagent cast"
```

---

### Task 7: `agent_deep.py` — chat multimodal, reflexión y paneles nuevos

**Files:**
- Modify: `agent_deep.py` (añadir celdas tras las de Task 6)
- Reference: `agent_full.py`

**Interfaces:**
- Consumes: todo lo de Tasks 5-6.
- Produces: `ejecutar_agente`, `render_artefacto`, paneles UI (skills, reparto, galería), celdas de inspección/dashboard.

- [ ] **Step 1: Celda de reflexión autónoma + `ejecutar_agente` multimodal**

Copiar `agent_full.py:1137-1346` (clases `OperacionMemoria`/`SalidaReflexion`, `_ejecutar_reflexion_autonoma`, `ejecutar_agente`) con estos cambios en `ejecutar_agente`:

1. Añadir `DIR_ARTEFACTOS`, `ds`, `render_artefacto` a los parámetros de la celda (render_artefacto se define en Step 2 — su celda va ANTES en el archivo).
2. Antes de invocar el agente, tomar snapshot de artefactos; después, calcular nuevos:

```python
            # Snapshot para detectar artefactos creados durante el turno
            _antes = {p for p in DIR_ARTEFACTOS.rglob("*") if p.is_file()}

            salida = agente_cerebro.invoke(
                {"messages": [{"role": "user", "content": texto_usuario}]},
                cfg,
            )
```

3. Tras extraer `respuesta` (igual que el original), construir la vista multimodal y usarla en TODOS los yields finales en lugar del string plano:

```python
            _nuevos = sorted(
                {p for p in DIR_ARTEFACTOS.rglob("*") if p.is_file()} - _antes
            )

            def _vista(texto: str):
                """Texto + artefactos nuevos renderizados inline en el chat."""
                if not _nuevos:
                    return texto
                bloques = [mo.md(texto)]
                for _ruta in _nuevos:
                    bloques.append(mo.md(f"**📎 {_ruta.name}**"))
                    bloques.append(render_artefacto(_ruta))
                return mo.vstack(bloques)
```

   - El yield provisional pasa a: `yield _vista(respuesta + "\n\n*(🧠 Actualizando memoria autónoma...)*")`
   - El yield con cambios de memoria: `yield _vista(respuesta + f"\n\n---\n**🧠 Memoria Autónoma — Cambios detectados:**\n{detalle}")`
   - El yield sin cambios: `yield _vista(respuesta)`

- [ ] **Step 2: Celda `render_artefacto` (va antes de la celda del Step 1 en el archivo)**

```python
@app.cell
def _(ds, json, mo):
    def render_artefacto(ruta):
        """Convierte un archivo de ./artefactos en el componente Marimo adecuado."""
        categoria = ds.clasificar_artefacto(ruta)
        try:
            if categoria == "imagen":
                return mo.image(str(ruta))
            if categoria == "pdf":
                return mo.pdf(src=str(ruta))
            if categoria == "video":
                return mo.video(str(ruta))
            if categoria == "audio":
                return mo.audio(str(ruta))
            if categoria == "tabla":
                import polars as pl

                df = (
                    pl.read_parquet(ruta)
                    if ruta.suffix.lower() == ".parquet"
                    else pl.read_csv(ruta)
                )
                return mo.ui.table(df, selection=None)
            if categoria == "json":
                return mo.tree(json.loads(ruta.read_text(encoding="utf-8")))
            if categoria == "texto":
                return mo.md(ruta.read_text(encoding="utf-8"))
            if categoria == "html":
                return mo.Html(ruta.read_text(encoding="utf-8"))
        except Exception as e:
            return mo.callout(mo.md(f"No pude renderizar `{ruta.name}`: {e}"), kind="warn")
        # Categoría 'otro' → botón de descarga
        return mo.download(data=ruta.read_bytes(), filename=ruta.name)

    return (render_artefacto,)
```

- [ ] **Step 3: Celdas de chat**

Copiar `agent_full.py:1349-1377` (markdown + `mo.ui.chat`) añadiendo prompts nuevos a la lista:

```python
            "Lista tus skills disponibles y explica cuándo usarías cada una.",
            "Instala la skill 'artifacts-builder' del marketplace.",
            "Delega en el investigador: estado del arte de agentes con skills en 2026.",
            "Genera un gráfico de barras con estos datos: [{\"x\": \"a\", \"y\": 3}, {\"x\": \"b\", \"y\": 7}]",
```

- [ ] **Step 4: Panel de Skills**

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧩 Panel de Skills (estándar agentskills.io)

    Skills instaladas en `./skills/` — el mismo formato `SKILL.md` que usan
    Claude Code y otros arneses. El agente ve solo los *frontmatter* al arrancar
    y lee el contenido completo únicamente cuando la tarea lo requiere
    (**progressive disclosure**). Instala desde el marketplace escribiendo un
    nombre corto (`pdf`), una URL de carpeta GitHub o una URL raw a un `SKILL.md`.
    """)
    return


@app.cell
def _(mo):
    ui_fuente_skill = mo.ui.text(
        placeholder="nombre corto · URL github.com/.../tree/... · URL raw SKILL.md",
        label="**Fuente de la skill**",
        full_width=True,
    )
    ui_boton_instalar = mo.ui.run_button(label="⬇️ Instalar skill")
    ui_boton_recargar_skills = mo.ui.run_button(label="🔄 Recargar")
    return ui_boton_instalar, ui_boton_recargar_skills, ui_fuente_skill


@app.cell
def _(
    DIR_SKILLS,
    ds,
    mo,
    ui_boton_instalar,
    ui_boton_recargar_skills,
    ui_fuente_skill,
):
    _ = ui_boton_recargar_skills.value  # dependencia reactiva

    _msg_instalacion = ""
    if ui_boton_instalar.value and ui_fuente_skill.value.strip():
        try:
            _msg_instalacion = ds.instalar_skill_desde_fuente(
                ui_fuente_skill.value.strip(), None, DIR_SKILLS
            )
        except Exception as _e:
            _msg_instalacion = f"❌ Error: {_e}"

    _skills, _avisos_skills = ds.listar_skills(DIR_SKILLS)

    _bloques = [
        mo.hstack(
            [ui_fuente_skill, ui_boton_instalar, ui_boton_recargar_skills],
            widths=[3, 1, 1],
        )
    ]
    if _msg_instalacion:
        _bloques.append(
            mo.callout(
                mo.md(_msg_instalacion),
                kind="success" if _msg_instalacion.startswith("✅") else "danger",
            )
        )
    for _aviso in _avisos_skills:
        _bloques.append(mo.callout(mo.md(_aviso), kind="warn"))
    if _skills:
        _bloques.append(
            mo.ui.table(
                [{"Skill": s["name"], "Descripción": s["description"]} for s in _skills],
                selection=None,
            )
        )
    else:
        _bloques.append(
            mo.callout(mo.md("*(Sin skills — instala una arriba)*"), kind="info")
        )

    mo.vstack(_bloques)
    return
```

Nota: tras instalar, el usuario reinicia el kernel o pulsa Recargar y reejecuta la celda del agente para que DeepAgents relea `skills/` (documentarlo en el markdown del panel).

- [ ] **Step 5: Panel Constructor de Subagentes ("Reparto de la Obra")**

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🎭 El Reparto de la Obra — Constructor de Subagentes

    Cada personaje es un archivo `./subagentes/<nombre>.md` (frontmatter YAML +
    persona), el mismo formato que los *agents* de Claude Code. El **director**
    (agente principal) lee las `description` y delega escenas vía la tool `task`.
    Guardar o eliminar un personaje **reconstruye el agente al instante**.
    """)
    return


@app.cell
def _(herramientas_totales, mo):
    ui_sub_nombre = mo.ui.text(label="**Nombre** (sin espacios)", placeholder="critico")
    ui_sub_descripcion = mo.ui.text(
        label="**Descripción** — el director la lee para decidir delegar",
        placeholder="Delega aquí revisiones de calidad de textos.",
        full_width=True,
    )
    ui_sub_persona = mo.ui.text_area(
        label="**Persona / system prompt del personaje**",
        placeholder="Eres un crítico literario implacable pero justo...",
        rows=6,
        full_width=True,
    )
    ui_sub_tools = mo.ui.multiselect(
        options=[t.name for t in herramientas_totales],
        label="**Tools del personaje**",
    )
    ui_sub_modelo = mo.ui.dropdown(
        options={"Heredar del director": "", "Estándar": "estandar", "Razonamiento": "razonamiento"},
        value="Heredar del director",
        label="**Modelo**",
    )
    ui_boton_guardar_sub = mo.ui.run_button(label="💾 Guardar personaje")
    ui_sub_eliminar = mo.ui.text(label="**Eliminar personaje** (nombre)", placeholder="critico")
    ui_boton_eliminar_sub = mo.ui.run_button(label="🗑️ Eliminar")
    return (
        ui_boton_eliminar_sub,
        ui_boton_guardar_sub,
        ui_sub_descripcion,
        ui_sub_eliminar,
        ui_sub_modelo,
        ui_sub_nombre,
        ui_sub_persona,
        ui_sub_tools,
    )


@app.cell
def _(
    DIR_SUBAGENTES,
    avisos_subagentes,
    ds,
    marcar_version_reparto,
    mo,
    obtener_version_reparto,
    subagentes_cargados,
    ui_boton_eliminar_sub,
    ui_boton_guardar_sub,
    ui_sub_descripcion,
    ui_sub_eliminar,
    ui_sub_modelo,
    ui_sub_nombre,
    ui_sub_persona,
    ui_sub_tools,
):
    _msg_reparto = ""
    if ui_boton_guardar_sub.value:
        if ui_sub_nombre.value.strip() and ui_sub_persona.value.strip():
            ds.guardar_subagente_md(
                DIR_SUBAGENTES,
                name=ui_sub_nombre.value.strip(),
                description=ui_sub_descripcion.value.strip() or ui_sub_nombre.value,
                persona=ui_sub_persona.value,
                tools=list(ui_sub_tools.value),
                model=ui_sub_modelo.value or None,
            )
            marcar_version_reparto(obtener_version_reparto() + 1)
            _msg_reparto = f"✅ Personaje '{ui_sub_nombre.value}' guardado — agente reconstruido."
        else:
            _msg_reparto = "❌ Nombre y persona son obligatorios."

    if ui_boton_eliminar_sub.value and ui_sub_eliminar.value.strip():
        if ds.eliminar_subagente_md(DIR_SUBAGENTES, ui_sub_eliminar.value.strip()):
            marcar_version_reparto(obtener_version_reparto() + 1)
            _msg_reparto = f"🗑️ Personaje '{ui_sub_eliminar.value}' eliminado."
        else:
            _msg_reparto = f"❌ No existe '{ui_sub_eliminar.value}'."

    _formulario = mo.vstack(
        [
            mo.hstack([ui_sub_nombre, ui_sub_modelo], widths=[1, 1]),
            ui_sub_descripcion,
            ui_sub_persona,
            ui_sub_tools,
            mo.hstack(
                [ui_boton_guardar_sub, ui_sub_eliminar, ui_boton_eliminar_sub],
                widths=[1, 2, 1],
            ),
        ]
    )

    _bloques = [_formulario]
    if _msg_reparto:
        _bloques.append(
            mo.callout(
                mo.md(_msg_reparto),
                kind="danger" if _msg_reparto.startswith("❌") else "success",
            )
        )
    for _aviso in avisos_subagentes:
        _bloques.append(mo.callout(mo.md(_aviso), kind="warn"))
    if subagentes_cargados:
        _bloques.append(
            mo.ui.table(
                [
                    {
                        "Personaje": s["name"],
                        "Rol (description)": s["description"],
                        "Tools": ", ".join(t.name for t in s["tools"]) or "—",
                        "Modelo": "propio" if "model" in s else "heredado",
                    }
                    for s in subagentes_cargados
                ],
                selection=None,
            )
        )
    else:
        _bloques.append(mo.callout(mo.md("*(Reparto vacío)*"), kind="info"))

    mo.vstack(_bloques)
    return
```

- [ ] **Step 6: Galería de Artefactos**

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🖼️ Galería de Artefactos Multimodales

    Todo lo no-textual que produzcan las tools o el agente (vía `write_file`)
    aterriza en `./artefactos/` y se renderiza aquí según su tipo:
    imagen, PDF, video, audio, tabla, JSON, markdown, HTML o descarga directa.
    """)
    return


@app.cell
def _(mo):
    ui_boton_refrescar_galeria = mo.ui.run_button(label="🔄 Refrescar galería")
    ui_limite_galeria = mo.ui.slider(
        start=5, stop=50, step=5, value=10, label="**Máx. artefactos**"
    )
    return ui_boton_refrescar_galeria, ui_limite_galeria


@app.cell
def _(
    DIR_ARTEFACTOS,
    datetime,
    ds,
    mo,
    render_artefacto,
    ui_boton_refrescar_galeria,
    ui_limite_galeria,
):
    _ = ui_boton_refrescar_galeria.value  # dependencia reactiva

    _artefactos = ds.listar_artefactos(DIR_ARTEFACTOS, ui_limite_galeria.value)

    _cabecera = mo.hstack(
        [ui_boton_refrescar_galeria, ui_limite_galeria], widths=[1, 2]
    )
    if _artefactos:
        _acordeon = mo.accordion(
            {
                (
                    f"{_p.name} · {_p.stat().st_size / 1024:.1f} KB · "
                    f"{datetime.datetime.fromtimestamp(_p.stat().st_mtime):%Y-%m-%d %H:%M}"
                ): render_artefacto(_p)
                for _p in _artefactos
            }
        )
        mo.vstack([_cabecera, _acordeon])
    else:
        mo.vstack(
            [
                _cabecera,
                mo.callout(
                    mo.md("*(Sin artefactos — pide al agente un gráfico para probar)*"),
                    kind="info",
                ),
            ]
        )
    return
```

- [ ] **Step 7: Paneles restantes (paridad)**

Copiar verbatim de `agent_full.py`:
- Inspector de memoria: líneas 1379-1420.
- Visor de inyección dinámica: líneas 1423-1504 — CAMBIO: el panel del system prompt ahora muestra `PERSONAJE_BASE + sección de recuerdos` con nota de que DeepAgents antepone su andamiaje.
- Dashboard de estado: líneas 1507-1594 — CAMBIOS: añadir al markdown una fila/tabla con `len(subagentes_cargados)` personajes, número de skills (`len(ds.listar_skills(DIR_SKILLS)[0])`) y ruta de `DIR_ARTEFACTOS` (añadir esas variables a los parámetros de la celda).
- Editor de herramientas del estudiante: líneas 1602-1727 — copiar el editor y la celda de compilación; OMITIR las celdas duplicadas/muertas de `agent_full.py:1619-1695` (search_arxiv duplicado, `busqueda_arxiv`, `read_excel` con `pd` indefinido).
- Cerrar con `if __name__ == "__main__": app.run()` (líneas 1730-1731).

- [ ] **Step 8: Verificar sintaxis y commit**

Run: `uv run --no-project python -c "import ast; ast.parse(open('agent_deep.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

```bash
git add agent_deep.py
git commit -m "feat: add multimodal chat, skills panel, subagent builder, and gallery"
```

---

### Task 8: Verificación integral

**Files:**
- Modify (si hay fallos): `agent_deep.py`, `deep_soporte.py`

- [ ] **Step 1: Suite completa de tests**

Run: `uv run --with pyyaml,pytest pytest tests/test_deep_soporte.py -v`
Expected: 21 passed.

- [ ] **Step 2: Ejecución headless del notebook**

Marimo ejecuta todas las celdas al correr el script; sin `NVIDIA_API_KEY` el agente queda inactivo pero NINGUNA celda debe lanzar excepción.

Run: `uv run agent_deep.py`
Expected: exit 0, sin traceback. (Primera ejecución descarga deps del header uv — puede tardar.)

Si falla: leer el traceback, arreglar la celda correspondiente, repetir. Errores típicos: variable no devuelta en el `return` de una celda, nombre duplicado entre celdas (Marimo lo prohíbe), import faltante en los parámetros de la celda.

- [ ] **Step 3: Verificación manual de arranque UI (opcional si hay API key)**

Run: `uv run marimo edit agent_deep.py --headless --port 2718` (Ctrl+C tras comprobar que arranca)
Expected: servidor arranca sin errores en consola.

- [ ] **Step 4: Lint**

Run: `uv run --with ruff ruff check agent_deep.py deep_soporte.py tests/`
Expected: sin errores (o solo avisos de estilo aceptables en notebooks; arreglar los `F` de verdad).

- [ ] **Step 5: Commit final**

```bash
git add -A -- agent_deep.py deep_soporte.py tests/ skills/ subagentes/
git commit -m "feat: verified agent_deep notebook end to end"
```
