# ═══════════════════════════════════════════════════════════════════════════════════════
#  SERVIDOR_MCP · El "periférico" que enchufamos al Agente Profundo
# ═══════════════════════════════════════════════════════════════════════════════════════
#
#  Este archivo ES un servidor MCP completo. Léelo entero: son ~70 líneas.
#
#  · FastMCP (del SDK oficial `mcp`) convierte funciones Python en tools MCP,
#    igual que @tool de LangChain — pero aquí las tools viven en OTRO PROCESO
#    y se anuncian por un protocolo estándar (JSON-RPC sobre stdio).
#  · Cualquier cliente MCP (este notebook, Claude Code, Cursor...) puede
#    conectarse a este mismo archivo sin cambiar ni una línea.
#  · Ejercicio: añade tu propia @mcp.tool() abajo, pulsa "Recargar" en el
#    Panel MCP del notebook, y observa cómo el agente la descubre.
#
#  Ejecutar a mano (lo normal es que lo lance el notebook):
#      python servidor_mcp.py

from pathlib import Path

from mcp.server.fastmcp import FastMCP

RAIZ = Path(__file__).parent.resolve()

mcp = FastMCP("laboratorio-curso")

GLOSARIO = {
    "agente": "Programa que combina un LLM con tools en un ciclo ReAct: razonar, actuar, observar, repetir.",
    "tool": "Función Python que el LLM puede invocar; su docstring y type hints son el contrato.",
    "middleware": "Capa que envuelve al agente e intercepta petición/respuesta (límites, reintentos, PII...).",
    "skill": "Carpeta con SKILL.md (estándar agentskills.io) que el agente lee bajo demanda.",
    "subagente": "Personaje con contexto propio al que el director delega vía la tool task.",
    "mcp": "Model Context Protocol: el 'puerto USB-C' que conecta agentes con servidores de tools externos.",
    "checkpointer": "Persistencia del hilo de conversación turno a turno (memoria de corto plazo).",
    "store": "Base de hechos duraderos del usuario, compartida entre sesiones (memoria de largo plazo).",
}


@mcp.tool()
def consultar_glosario(termino: str) -> str:
    """Devuelve la definición de un término del curso de agentes de IA.

    Términos disponibles: agente, tool, middleware, skill, subagente, mcp,
    checkpointer, store.
    """
    clave = termino.lower().strip()
    if clave in GLOSARIO:
        return f"📖 {clave}: {GLOSARIO[clave]}"
    disponibles = ", ".join(sorted(GLOSARIO))
    return f"No tengo '{termino}'. Prueba con: {disponibles}"


@mcp.tool()
def estadisticas_curso() -> str:
    """Cuenta las skills, subagentes y artefactos instalados en este proyecto."""
    n_skills = len(list((RAIZ / "skills").glob("*/SKILL.md")))
    n_subagentes = len(list((RAIZ / "subagentes").glob("*.md")))
    n_artefactos = sum(1 for p in (RAIZ / "artefactos").rglob("*") if p.is_file())
    return (
        f"📊 Proyecto del curso: {n_skills} skills, "
        f"{n_subagentes} subagentes, {n_artefactos} artefactos."
    )


@mcp.tool()
def convertir_unidades(valor: float, de: str, a: str) -> str:
    """Convierte unidades comunes: km↔mi, kg↔lb, c↔f (Celsius/Fahrenheit)."""
    par = (de.lower().strip(), a.lower().strip())
    lineales = {
        ("km", "mi"): 0.621371,
        ("mi", "km"): 1.609344,
        ("kg", "lb"): 2.204623,
        ("lb", "kg"): 0.453592,
    }
    if par in lineales:
        return f"{valor} {par[0]} = {valor * lineales[par]:.4f} {par[1]}"
    if par == ("c", "f"):
        return f"{valor} °C = {valor * 9 / 5 + 32:.2f} °F"
    if par == ("f", "c"):
        return f"{valor} °F = {(valor - 32) * 5 / 9:.2f} °C"
    return f"No sé convertir de '{de}' a '{a}'. Pares: km/mi, kg/lb, c/f."


@mcp.resource("curso://glosario")
def glosario_completo() -> str:
    """Glosario completo del curso en markdown (ejemplo de RESOURCE MCP:
    contenido de solo lectura que un cliente puede pedir, distinto de una tool)."""
    return "\n".join(f"- **{k}**: {v}" for k, v in sorted(GLOSARIO.items()))


if __name__ == "__main__":
    # Sin argumentos = transporte stdio: el proceso lee JSON-RPC por stdin
    # y responde por stdout. Por eso NUNCA se imprime nada a stdout aquí.
    mcp.run()
