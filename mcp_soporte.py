# ═══════════════════════════════════════════════════════════════════════════════════════
#  MCP_SOPORTE · Lógica pura para la integración MCP del Agente Profundo
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  Este módulo NO importa marimo ni langchain: solo stdlib. Así todo se prueba
#  con pytest sin levantar el notebook ni tocar la red.
#
#  El archivo mcp_config.json usa el MISMO formato que claude_desktop_config.json
#  (el estándar de facto del ecosistema MCP): {"mcpServers": {nombre: {...}}}.

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from deep_soporte import _nombre_seguro

TRANSPORTES_VALIDOS = {"stdio", "http", "sse"}

SEMILLA_MCP = {
    "laboratorio": {
        "transport": "stdio",
        "command": "python",
        "args": ["servidor_mcp.py"],
        "enabled": True,
    }
}


def cargar_config_mcp(ruta: Path) -> tuple[dict, list[str]]:
    """Lee mcp_config.json → (servidores, avisos). Nunca lanza excepción."""
    ruta = Path(ruta)
    if not ruta.exists():
        return {}, []
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {}, [f"⚠️ mcp_config.json ilegible: {e}"]
    servidores = datos.get("mcpServers", {})
    if not isinstance(servidores, dict):
        return {}, ["⚠️ 'mcpServers' debe ser un objeto JSON {nombre: config}"]
    return servidores, []


def guardar_config_mcp(ruta: Path, servidores: dict) -> None:
    """Escribe el dict de servidores con el envoltorio {"mcpServers": ...}."""
    Path(ruta).write_text(
        json.dumps({"mcpServers": servidores}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def validar_servidor(cfg: dict) -> list[str]:
    """Devuelve la lista de errores de una config de servidor (vacía = válida)."""
    errores = []
    transporte = cfg.get("transport", "stdio")
    if transporte not in TRANSPORTES_VALIDOS:
        errores.append(f"transporte inválido: {transporte!r} (usa stdio/http/sse)")
    if transporte == "stdio" and not cfg.get("command"):
        errores.append("el transporte stdio requiere 'command'")
    if transporte in ("http", "sse") and not cfg.get("url"):
        errores.append(f"el transporte {transporte} requiere 'url'")
    return errores


def agregar_servidor(ruta: Path, nombre: str, cfg: dict) -> str:
    """Valida y persiste un servidor. Lanza ValueError si nombre/config inválidos."""
    nombre = _nombre_seguro(nombre)
    errores = validar_servidor(cfg)
    if errores:
        raise ValueError("; ".join(errores))
    servidores, _ = cargar_config_mcp(ruta)
    servidores[nombre] = cfg
    guardar_config_mcp(ruta, servidores)
    return f"✅ Servidor '{nombre}' guardado en {Path(ruta).name}"


def eliminar_servidor(ruta: Path, nombre: str) -> bool:
    """Elimina un servidor del JSON. True si existía."""
    nombre = _nombre_seguro(nombre)
    servidores, _ = cargar_config_mcp(ruta)
    if nombre not in servidores:
        return False
    del servidores[nombre]
    guardar_config_mcp(ruta, servidores)
    return True


def parsear_env(texto: str) -> dict:
    """Convierte líneas 'CLAVE=valor' en dict. Ignora comentarios y ruido."""
    env = {}
    for linea in (texto or "").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave = clave.strip()
        if clave:
            env[clave] = valor.strip()
    return env


def normalizar_conexiones(servidores: dict, ejecutable: str, raiz: Path) -> dict:
    """Del JSON crudo al formato que espera MultiServerMCPClient.

    - Filtra servidores con "enabled": false (deshabilitar sin borrar).
    - "command": "python" → el Python del venv actual (sys.executable del
      notebook): evita que el subprocess use otro intérprete sin las deps.
    - Args relativos que apuntan a archivos existentes bajo `raiz` → absolutos
      (el cwd del subprocess no está garantizado).
    - Quita la clave 'enabled' (no es parte del contrato del cliente).
    """
    conexiones = {}
    raiz = Path(raiz)
    for nombre, cfg in servidores.items():
        if not isinstance(cfg, dict) or cfg.get("enabled", True) is False:
            continue
        limpio = {k: v for k, v in cfg.items() if k != "enabled"}
        limpio.setdefault("transport", "stdio")
        if limpio.get("command") == "python":
            limpio["command"] = ejecutable
        args = limpio.get("args")
        if isinstance(args, list):
            limpio["args"] = [
                str((raiz / a).resolve())
                if isinstance(a, str)
                and not Path(a).is_absolute()
                and (raiz / a).is_file()
                else a
                for a in args
            ]
        conexiones[nombre] = limpio
    return conexiones


def sembrar_config_mcp(ruta: Path) -> None:
    """Crea mcp_config.json con el servidor demo si no existe (idempotente)."""
    ruta = Path(ruta)
    if not ruta.exists():
        guardar_config_mcp(ruta, dict(SEMILLA_MCP))


def dialogo_crudo_stdio(
    comando: list[str], peticiones: list[dict], timeout: float = 20.0
) -> list[dict]:
    """Habla JSON-RPC CRUDO con un servidor MCP stdio y devuelve sus respuestas.

    ⚠️ SOLO EDUCATIVO — la celda "radiografía" lo usa para enseñar el cable del
    protocolo (un mensaje JSON por línea sobre stdin/stdout). En producción el
    agente usa langchain-mcp-adapters, que gestiona sesiones, reintentos y
    versiones por nosotros.

    Envía todas las peticiones y MANTIENE stdin ABIERTO hasta recibir todas las
    respuestas esperadas (una por cada petición con "id"). Esto es clave: un
    servidor MCP stdio interpreta el EOF de stdin como orden de apagado y
    CANCELA cualquier petición en vuelo — típicamente la última y más pesada
    (p. ej. un tools/call), cuya respuesta se perdería si cerrásemos stdin de
    inmediato. Por eso un hilo lector recolecta stdout mientras stdin sigue
    abierto, y sólo matamos el proceso cuando ya llegó todo (o vence el timeout).

    Las notificaciones (mensajes sin "id") no generan respuesta — por eso el
    resultado puede ser más corto que la entrada.

    Contrato:
    - Devuelve las respuestas en el ORDEN DE LLEGADA.
    - Si llegaron ALGUNAS pero no todas las esperadas dentro del timeout,
      devuelve la lista PARCIAL (el llamador dibuja las que faltan).
    - Lanza TimeoutError SÓLO cuando no llegó NADA.
    """
    esperadas = {p["id"] for p in peticiones if "id" in p}
    proc = subprocess.Popen(
        comando,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
    )
    respuestas: list[dict] = []

    def _leer():
        for linea in proc.stdout:
            try:
                msg = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "id" in msg:
                respuestas.append(msg)
                if {m["id"] for m in respuestas} >= esperadas:
                    break

    hilo = threading.Thread(target=_leer, daemon=True)
    hilo.start()
    try:
        for p in peticiones:
            proc.stdin.write(json.dumps(p) + "\n")
        proc.stdin.flush()
        hilo.join(timeout)
    finally:
        proc.kill()
        proc.wait()  # reap: sin zombies (corrige también un minor previo)
    if not respuestas:
        raise TimeoutError(
            f"El servidor MCP no respondió en {timeout}s: {comando}"
        )
    return respuestas
