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

_PATRON_NOMBRE_SEGURO = re.compile(r"^[A-Za-z0-9._-]+$")


def _nombre_seguro(nombre: str) -> str:
    """Valida un componente de ruta: sin separadores, sin '..', solo [A-Za-z0-9._-].

    Lanza ValueError si el nombre podría escapar del directorio destino.
    """
    if not _PATRON_NOMBRE_SEGURO.match(nombre) or nombre in {".", ".."}:
        raise ValueError(f"Nombre de archivo/carpeta inseguro: {nombre!r}")
    return nombre


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
        nombre_entrada = _nombre_seguro(entrada["name"])
        if entrada["type"] == "dir":
            _descargar_carpeta_github(
                owner, repo, entrada["path"], ref,
                destino / nombre_entrada, descargar, _contador,
            )
        elif entrada["type"] == "file":
            if entrada.get("size", 0) > MAX_BYTES:
                continue
            (destino / nombre_entrada).write_bytes(
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
        try:
            nombre_final = _nombre_seguro(nombre or fuente.rstrip("/").split("/")[-2])
        except ValueError as e:
            return f"❌ {e}"
        destino = dir_skills / nombre_final
        destino.mkdir(parents=True, exist_ok=True)
        (destino / "SKILL.md").write_bytes(descargar(fuente))
        return f"✅ Skill '{nombre_final}' instalada (1 archivo) en {destino}"

    # Caso 2 · URL de carpeta github.com/.../tree/<ref>/<ruta>
    m = _PATRON_TREE_GITHUB.match(fuente)
    if m:
        owner, repo, ref, ruta = m.groups()
        try:
            nombre_final = _nombre_seguro(nombre or ruta.rstrip("/").split("/")[-1])
            n = _descargar_carpeta_github(
                owner, repo, ruta, ref, dir_skills / nombre_final, descargar
            )
        except ValueError as e:
            return f"❌ {e}"
        return f"✅ Skill '{nombre_final}' instalada ({n} archivos)"

    # Caso 3 · Nombre corto → sondear repos conocidos del marketplace
    if not fuente.startswith("http"):
        try:
            nombre_final = _nombre_seguro(nombre or fuente.split("/")[-1])
        except ValueError as e:
            return f"❌ {e}"
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
        nombres_tools = fm.get("tools") or []
        if isinstance(nombres_tools, str):  # YAML escalar → lista de uno
            nombres_tools = [nombres_tools]
        for nombre_tool in nombres_tools:
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
    """Escribe <directorio>/<name>.md con frontmatter + persona.

    Lanza ValueError si `name` podría escapar del directorio destino
    (mismo criterio que `_nombre_seguro`, usado en el instalador de skills).
    """
    _nombre_seguro(name)
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
    """Borra el archivo del personaje. True si existía.

    Lanza ValueError si `name` podría escapar del directorio destino.
    """
    _nombre_seguro(name)
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
