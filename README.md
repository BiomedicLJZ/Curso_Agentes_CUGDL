# Agente01

Aplicación de consola en Python que conecta un agente conversacional con **NVIDIA NIM** mediante `langchain-nvidia-ai-endpoints` y muestra tanto la respuesta final como el razonamiento del modelo con una interfaz enriquecida usando **Rich**.

## Características

- Chat interactivo en terminal.
- Historial de conversación dentro de la sesión.
- Prompt de sistema en español con respuestas formales y concisas.
- Renderizado visual del razonamiento y la respuesta final.
- Integración con el modelo `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.

## Requisitos

- Python 3.13 o superior.
- Una clave de API de NVIDIA disponible en la variable de entorno `NVIDIA_API_KEY`.

## Instalación

Este proyecto usa `uv` para gestionar dependencias.

```powershell
uv sync
```

Si prefieres `pip`, puedes instalar las dependencias definidas en `pyproject.toml`.

## Configuración

Antes de ejecutar la aplicación, define tu clave de API:

```powershell
$env:NVIDIA_API_KEY="tu_llave_api"
```

> Nota: aunque el código incluye un valor por defecto de marcador, debes reemplazarlo con una clave real en la variable de entorno para usar el agente correctamente.

## Uso

Ejecuta la aplicación con:

```powershell
uv run python .\main.py
```

Durante la sesión:

- Escribe tu pregunta y presiona Enter.
- Usa `salir`, `exit` o `quit` para terminar.

## Estructura principal

- `main.py`: inicializa el modelo, mantiene el historial y renderiza la salida.
- `pyproject.toml`: define metadatos y dependencias del proyecto.

## Dependencias principales

- `langchain`
- `langchain-nvidia-ai-endpoints`
- `rich`
- `pylatexenc`