import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# ─── Console global ──────────────────────────────────────────────────────────
console = Console()

# ─── Conexión con el LLM de NVIDIA ───────────────────────────────────────────
llm = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    api_key=os.getenv("NVIDIA_API_KEY", "#POR FAVOR PONER AQUI SU PROPIA LLAVE API"),
    temperature=0.6,
    top_p=0.95,
    max_tokens=65536,
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
)

SYSTEM_PROMPT = (
    "Eres un asistente de IA diseñado para resolver las dudas que te hagan "
    "los usuarios. Responde con un tono formal y conciso."
)


# ─── Pretty Printer ───────────────────────────────────────────────────────────
def pretty_print_response(message: AIMessage) -> None:
    """
    Renders an AIMessage from a reasoning model.
    Handles two possible content formats:
      1. content_blocks list  → [{'type':'thinking','thinking':'...'}, {'type':'text','text':'...'}]
      2. Plain string content → '<think>...</think>respuesta...'
    """
    thinking_text = None
    response_text = None

    # ── Caso 1: el modelo devuelve content_blocks estructurados ──
    raw_blocks = getattr(message, "content_blocks", None)
    if raw_blocks:
        for block in raw_blocks:
            btype = getattr(block, "type", None) or block.get("type")
            if btype == "thinking":
                thinking_text = getattr(block, "thinking", None) or block.get(
                    "thinking", ""
                )
            elif btype == "text":
                response_text = getattr(block, "text", None) or block.get("text", "")

    # ── Caso 2: el modelo devuelve string con tags <think> ──
    if response_text is None:
        raw = message.content if isinstance(message.content, str) else ""
        if "</think>" in raw:
            parts = raw.split("</think>", 1)
            thinking_text = parts[0].replace("<think>", "").strip()
            response_text = parts[1].strip()
        else:
            response_text = raw.strip()

    # ── Render ──────────────────────────────────────────────────
    console.print()

    # Bloque de razonamiento (colapsado visualmente, dim)
    if thinking_text:
        console.print(
            Panel(
                Text(thinking_text, style="dim italic"),
                title="[bold yellow]⚙  Razonamiento interno[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                expand=False,
                padding=(1, 2),
            )
        )
        console.print()

    # Respuesta final
    console.print(
        Panel(
            Markdown(response_text),
            title="[bold cyan]🤖  Asistente[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print()


# ─── Loop principal ───────────────────────────────────────────────────────────
def main():
    historial: list = [SystemMessage(content=SYSTEM_PROMPT)]

    console.print(
        Rule("[bold green]Agente QA — NVIDIA Nemotron Reasoning[/bold green]")
    )
    console.print("[dim]Escribe 'salir' para terminar.[/dim]\n")

    while True:
        try:
            mensaje = console.input("[bold green]Tú:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Sesión interrumpida.[/dim]")
            break

        if not mensaje:
            continue

        if mensaje.lower() in ("salir", "exit", "quit"):
            console.print("[dim]Sesión terminada.[/dim]")
            break

        historial.append(HumanMessage(content=mensaje))

        with console.status("[bold yellow]Pensando...[/bold yellow]", spinner="dots"):
            respuesta: AIMessage = llm.invoke(historial)

        pretty_print_response(respuesta)

        # Guardamos solo el texto limpio en el historial
        clean_content = respuesta.content
        if "</think>" in clean_content:
            clean_content = clean_content.split("</think>")[-1].strip()
        historial.append(AIMessage(content=clean_content))


if __name__ == "__main__":
    main()
