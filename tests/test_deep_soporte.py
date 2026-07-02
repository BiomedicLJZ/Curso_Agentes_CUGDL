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


# ── instalar_skill_desde_fuente ──────────────────────────────────────────────

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
