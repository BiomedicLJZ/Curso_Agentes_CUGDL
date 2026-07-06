# crisol_v1.py — Acto 1: Subagentes en producción
from langchain.agents.middleware import PIIMiddleware
import os
from typing import Literal
from tavily import TavilyClient
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2)
model2 = ChatNVIDIA(model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",temperature=0.6)

def buscar_literatura(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
):
    """Busca fuentes y devuelve resultados con URL y snippet."""
    return tavily_client.search(query, max_results=max_results, topic=topic)

buscador = {
    "name": "buscador",
    "description": (
        "Investiga preguntas académicas con fuentes externas. "
        "NUNCA redacta, solo investiga."
    ),
    "system_prompt": (
        "Eres un investigador riguroso. Por cada afirmación, incluye la URL "
        "de la fuente. Si la evidencia es débil, dilo explícitamente. "
        "Nunca inventes hallazgos ni fuentes que no vengan de tu herramienta."
    ),
    "tools": [buscar_literatura],
    "model": model2
}

redactor = {
    "name": "redactor",
    "description": (
        "Redacta secciones académicas A PARTIR de hallazgos YA verificados "
        "que se le entreguen explícitamente. No investiga por su cuenta."
    ),
    "system_prompt": (
        "Eres un redactor técnico. SOLO citas hallazgos presentes en el "
        "mensaje que recibiste. Si el orquestador no te dio fuente para una "
        "afirmación, no la incluyas. Cada dato lleva su URL entre paréntesis."
    ),
    "tools": [],
    "model": model2
}

crisol = create_deep_agent(
    model=model,
    system_prompt=(
        "Eres CRISOL, orquestador de un sistema de apoyo a investigación "
        "docente. No investigas ni redactas directamente: delegas. "
        "1) Envía la pregunta al subagente 'buscador'. "
        "2) Con hallazgos verificados, delega la redacción al 'redactor', "
        "   pasándole EXPLÍCITAMENTE cada hallazgo con su fuente. "
        "3) Entrega el resultado final."
    ),
    subagents=[buscador, redactor],
)

if __name__ == "__main__":
    resultado = crisol.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "¿Qué evidencia hay sobre el efecto de la retroalimentación "
                "inmediata en el aprendizaje de programación introductoria?"
            ),
        }]
    })
    print(resultado["messages"][-1].content)
