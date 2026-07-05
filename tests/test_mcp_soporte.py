"""Tests de mcp_soporte.py — config MCP pura, sin red ni subprocess."""

import json

import pytest

import mcp_soporte as ms


# ── cargar_config_mcp ────────────────────────────────────────────────────────


def test_cargar_config_inexistente_devuelve_vacio(tmp_path):
    servidores, avisos = ms.cargar_config_mcp(tmp_path / "no_existe.json")
    assert servidores == {}
    assert avisos == []


def test_cargar_config_corrupta_devuelve_aviso(tmp_path):
    ruta = tmp_path / "mcp_config.json"
    ruta.write_text("{esto no es json", encoding="utf-8")
    servidores, avisos = ms.cargar_config_mcp(ruta)
    assert servidores == {}
    assert len(avisos) == 1
    assert "ilegible" in avisos[0]


def test_cargar_config_mcpservers_no_dict_devuelve_aviso(tmp_path):
    ruta = tmp_path / "mcp_config.json"
    ruta.write_text(json.dumps({"mcpServers": [1, 2]}), encoding="utf-8")
    servidores, avisos = ms.cargar_config_mcp(ruta)
    assert servidores == {}
    assert len(avisos) == 1


# ── agregar / eliminar / roundtrip ───────────────────────────────────────────


def test_agregar_y_recargar_roundtrip(tmp_path):
    ruta = tmp_path / "mcp_config.json"
    msg = ms.agregar_servidor(
        ruta, "docs", {"transport": "http", "url": "http://localhost:3000/mcp"}
    )
    assert msg.startswith("✅")
    servidores, _ = ms.cargar_config_mcp(ruta)
    assert servidores["docs"]["url"] == "http://localhost:3000/mcp"


def test_agregar_nombre_inseguro_lanza_valueerror(tmp_path):
    with pytest.raises(ValueError):
        ms.agregar_servidor(
            tmp_path / "c.json", "../evil", {"transport": "stdio", "command": "x"}
        )


def test_agregar_config_invalida_lanza_valueerror(tmp_path):
    with pytest.raises(ValueError):
        ms.agregar_servidor(tmp_path / "c.json", "roto", {"transport": "stdio"})


def test_eliminar_servidor_existente_e_inexistente(tmp_path):
    ruta = tmp_path / "c.json"
    ms.agregar_servidor(ruta, "temporal", {"transport": "stdio", "command": "x"})
    assert ms.eliminar_servidor(ruta, "temporal") is True
    assert ms.eliminar_servidor(ruta, "temporal") is False


# ── validar_servidor ─────────────────────────────────────────────────────────


def test_validar_transporte_invalido():
    assert ms.validar_servidor({"transport": "palomas"}) != []


def test_validar_stdio_requiere_command():
    assert ms.validar_servidor({"transport": "stdio"}) != []
    assert ms.validar_servidor({"transport": "stdio", "command": "python"}) == []


def test_validar_http_requiere_url():
    assert ms.validar_servidor({"transport": "http"}) != []
    assert ms.validar_servidor({"transport": "http", "url": "http://x/mcp"}) == []


# ── parsear_env ──────────────────────────────────────────────────────────────


def test_parsear_env_basico_y_ruido():
    texto = "API_KEY=abc123\n# comentario\n\nsin_igual\nOTRA = con espacios "
    env = ms.parsear_env(texto)
    assert env == {"API_KEY": "abc123", "OTRA": "con espacios"}


def test_parsear_env_vacio():
    assert ms.parsear_env("") == {}
    assert ms.parsear_env(None) == {}


# ── normalizar_conexiones ────────────────────────────────────────────────────


def test_normalizar_filtra_deshabilitados_y_quita_enabled(tmp_path):
    servidores = {
        "on": {"transport": "stdio", "command": "x", "enabled": True},
        "off": {"transport": "stdio", "command": "y", "enabled": False},
    }
    conexiones = ms.normalizar_conexiones(servidores, "/venv/python", tmp_path)
    assert "off" not in conexiones
    assert "enabled" not in conexiones["on"]


def test_normalizar_python_usa_ejecutable_del_venv(tmp_path):
    servidores = {"lab": {"transport": "stdio", "command": "python", "args": []}}
    conexiones = ms.normalizar_conexiones(servidores, "/venv/bin/python", tmp_path)
    assert conexiones["lab"]["command"] == "/venv/bin/python"


def test_normalizar_resuelve_args_relativos_existentes(tmp_path):
    (tmp_path / "servidor_mcp.py").write_text("# demo", encoding="utf-8")
    servidores = {
        "lab": {
            "transport": "stdio",
            "command": "python",
            "args": ["servidor_mcp.py", "--flag"],
        }
    }
    conexiones = ms.normalizar_conexiones(servidores, "py", tmp_path)
    assert conexiones["lab"]["args"][0] == str(
        (tmp_path / "servidor_mcp.py").resolve()
    )
    assert conexiones["lab"]["args"][1] == "--flag"  # no-archivo queda igual


def test_normalizar_default_transport_stdio(tmp_path):
    conexiones = ms.normalizar_conexiones(
        {"lab": {"command": "x"}}, "py", tmp_path
    )
    assert conexiones["lab"]["transport"] == "stdio"


# ── sembrar ──────────────────────────────────────────────────────────────────


def test_sembrar_config_idempotente(tmp_path):
    ruta = tmp_path / "mcp_config.json"
    ms.sembrar_config_mcp(ruta)
    servidores, _ = ms.cargar_config_mcp(ruta)
    assert "laboratorio" in servidores
    ms.agregar_servidor(ruta, "extra", {"transport": "stdio", "command": "x"})
    ms.sembrar_config_mcp(ruta)  # segunda llamada NO pisa
    servidores2, _ = ms.cargar_config_mcp(ruta)
    assert "extra" in servidores2
