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


# ── seguridad: path traversal en el instalador de skills ────────────────────


def test_instalar_carpeta_github_con_nombre_traversal_no_escribe_fuera(tmp_path):
    """Una entrada 'name': '../evil.py' en el listado de GitHub no debe poder
    escribir fuera de la carpeta destino de la skill."""
    api = "https://api.github.com/repos/own/rep/contents/skills/bar?ref=main"
    listado = [
        {
            "type": "file",
            "name": "SKILL.md",
            "size": 100,
            "download_url": "https://raw.example/SKILL.md",
        },
        {
            "type": "file",
            "name": "../evil.py",
            "size": 10,
            "download_url": "https://raw.example/evil.py",
        },
    ]
    fake = _fake_descargar_factory(
        {
            api: json.dumps(listado).encode(),
            "https://raw.example/SKILL.md": FM_VALIDO.encode(),
            "https://raw.example/evil.py": b"import os; os.system('rm -rf /')",
        }
    )

    ds.instalar_skill_desde_fuente(
        "https://github.com/own/rep/tree/main/skills/bar", None, tmp_path, descargar=fake
    )

    # Nada debe escribirse fuera de tmp_path (la carpeta que contiene todo el
    # árbol de destino usado en el test).
    assert not (tmp_path / "evil.py").exists()
    assert not (tmp_path.parent / "evil.py").exists()
    for ruta in tmp_path.rglob("evil.py"):
        raise AssertionError(f"Archivo malicioso escrito en {ruta}")


def test_instalar_carpeta_github_con_nombre_dir_traversal_no_escapa(tmp_path):
    """Una entrada 'type': 'dir', 'name': '..' no debe permitir recursión hacia
    fuera de la carpeta destino."""
    api = "https://api.github.com/repos/own/rep/contents/skills/bar?ref=main"
    listado = [
        {
            "type": "file",
            "name": "SKILL.md",
            "size": 100,
            "download_url": "https://raw.example/SKILL.md",
        },
        {"type": "dir", "name": "..", "path": "skills"},
    ]
    fake = _fake_descargar_factory(
        {
            api: json.dumps(listado).encode(),
            "https://raw.example/SKILL.md": FM_VALIDO.encode(),
        }
    )

    ds.instalar_skill_desde_fuente(
        "https://github.com/own/rep/tree/main/skills/bar", None, tmp_path, descargar=fake
    )

    # No debe haberse creado/escrito nada en tmp_path directamente (solo dentro
    # de tmp_path/bar debería existir cualquier archivo).
    assert not (tmp_path / "SKILL.md").exists()


def test_instalar_url_raw_con_segmento_traversal_retorna_error(tmp_path):
    """Una URL raw cuyo penúltimo segmento es '..' no debe derivar un nombre de
    carpeta que escape de dir_skills."""
    url = "https://raw.example/skills/../SKILL.md"
    fake = _fake_descargar_factory({url: FM_VALIDO.encode()})

    msg = ds.instalar_skill_desde_fuente(url, None, tmp_path, descargar=fake)

    assert msg.startswith("❌")
    assert not (tmp_path.parent / "SKILL.md").exists()
    # No debe haberse creado ninguna carpeta '..' resuelta fuera de tmp_path.
    assert list(tmp_path.iterdir()) == []


# ── subagentes ──────────────────────────────────────────────────────────────

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


def test_cargar_subagentes_tools_escalar(tmp_path):
    # YAML permite `tools: tool_a` (escalar) — debe tratarse como lista de uno
    (tmp_path / "solo.md").write_text(
        "---\nname: solo\ndescription: x\ntools: tool_a\n---\npersona",
        encoding="utf-8",
    )
    subs, avisos = ds.cargar_subagentes(tmp_path, {"tool_a": "OBJ"})
    assert avisos == []
    assert subs[0]["tools"] == ["OBJ"]
