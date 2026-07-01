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
