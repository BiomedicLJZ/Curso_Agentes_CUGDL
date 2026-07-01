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
