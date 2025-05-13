import os
import shutil
import subprocess
import requests
from ollama import Client
from .printer import Printer
from openai import OpenAI

printer = Printer("AI INTERFACE")

CONTEXT_DIR = os.getenv("CONTEXT_DIR", "server/ai/context")
FAQ_FILE_PATH = os.path.join(CONTEXT_DIR, "FAQ.txt")
SYSTEM_PROMPT_FILE_PATH = os.path.join(CONTEXT_DIR, "SYSTEM.txt")


def get_faq_questions() -> list[str]:
    """
    Lee las preguntas frecuentes desde el archivo.
    Lanza FileNotFoundError si no existe el archivo.
    """
    if not os.path.exists(FAQ_FILE_PATH):
        raise FileNotFoundError(f"Archivo de FAQ no encontrado: {FAQ_FILE_PATH}")

    with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_system_prompt() -> str:
    """
    Lee el prompt del sistema desde el archivo.
    Lanza FileNotFoundError si no existe el archivo.
    """

    # Try to get from the file
    if not os.path.exists(SYSTEM_PROMPT_FILE_PATH):
        raise FileNotFoundError(
            f"Archivo de prompt del sistema no encontrado: {SYSTEM_PROMPT_FILE_PATH}"
        )

    with open(SYSTEM_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_physical_context() -> str:
    context_files = os.listdir("server/ai/context")
    valid_extensions = (".md", ".txt", ".csv")
    ignored_files = {"SYSTEM.txt", "FAQ.txt"}

    context = ""
    for file in context_files:
        if file.endswith(valid_extensions) and file not in ignored_files:
            try:
                with open(f"server/ai/context/{file}", "r", encoding="utf-8") as f:
                    context += f'<FILE name="{file}" used_for="ai_context">\n'
                    context += f.read()
                    context += "</FILE>\n"
            except Exception as e:
                printer.red(f"Error reading file {file}: {e}")
    return context


DEFAULT_WARNING_TEXT = """⚠️ Aviso Importante:

El contenido mostrado, incluyendo: textos, gráficos, imágenes u otro tipo de material incluido en el sitio web denominado ‘Sentencia Ciudadana’, tiene exclusivamente una finalidad informativa de lectura simple. Por tanto, no debe ser entendido o concebido como un sustituto de la resolución judicial; en consecuencia, el texto mostrado no tiene ningún valor legal.

Este resumen fue generado automáticamente por inteligencia artificial para facilitar la comprensión general del/los adjunto(s). Puede contener errores u omisiones debido a la calidad del texto, del archivo original o a su complejidad.
"""


def get_warning_text():
    warning_text = os.getenv("WARNING_TEXT", DEFAULT_WARNING_TEXT)
    if not warning_text:
        return DEFAULT_WARNING_TEXT
    return warning_text


def check_ollama_installation() -> dict:
    result = {
        "installed": False,
        "server_running": False,
        "version": None,
        "error": None,
    }

    # Verificar si el binario existe
    if not shutil.which("ollama"):
        result["error"] = "Ollama no está instalado o no está en el PATH."
        return result

    result["installed"] = True

    # Verificar versión
    try:
        version_output = subprocess.check_output(
            ["ollama", "--version"], text=True
        ).strip()
        result["version"] = version_output
    except subprocess.CalledProcessError:
        result["error"] = "No se pudo obtener la versión de Ollama."
        return result

    # Verificar si el servidor está corriendo
    try:
        r = requests.get("http://localhost:11434")
        if r.status_code == 200:
            result["server_running"] = True
    except requests.ConnectionError:
        result["error"] = "Ollama está instalado pero el servidor no está corriendo."

    return result


class OllamaProvider:
    def __init__(self):
        self.client = Client()

    def check_model(self, model: str = "gemma3:1b"):
        """Verifica si el modelo está disponible; si no, lo descarga."""
        model_list = self.client.list()
        available = [m.model for m in model_list.models]
        if model not in available:
            print(f"Modelo '{model}' no encontrado. Descargando...")
            self.client.pull(model)
        else:
            print(f"Modelo '{model}' disponible.")

    def embed(self, text: str, model: str = "nomic-embed-text"):
        return self.client.embed(model=model, input=text)

    def chat(
        self,
        messages: list[dict],
        model: str = "gemma3:1b",
        stream: bool = False,
        tools: list[dict] | list[callable] = [],
    ):
        self.check_model(model)
        response = self.client.chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            options={"num_ctx": 50000},
        )
        return response.message.content


class OpenAIProvider:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        stream: bool = False,
        tools: list[dict] | list[callable] = [],
    ):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
        )
        # printer.yellow(response, "RESPONSE")

        return response.choices[0].message.content


class AIInterface:
    client: OllamaProvider | OpenAIProvider | None = None

    def __init__(self, provider: str = "ollama", api_key: str = None):
        self.provider = provider
        if provider == "ollama":
            self.client = OllamaProvider()
        elif provider == "openai":
            self.client = OpenAIProvider(api_key)
        else:
            raise ValueError(f"Provider {provider} not supported")

        printer.blue("Using AI from", self.provider)

    def embed(self, text: str, model: str = "nomic-embed-text"):
        return self.client.embed(text, model)

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = False,
        tools: list[dict] | list[callable] = [],
    ):
        return self.client.chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
        )
