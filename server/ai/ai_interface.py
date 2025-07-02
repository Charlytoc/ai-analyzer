import os
import shutil
import json
import uuid
import subprocess
from functools import lru_cache
import requests
from ollama import Client
from ..utils.printer import Printer
from openai import OpenAI

printer = Printer("AI INTERFACE")

CONTEXT_DIR = os.getenv("CONTEXT_DIR", "server/ai/context")
FAQ_FILE_PATH = os.path.join(CONTEXT_DIR, "FAQ.txt")
SYSTEM_PROMPT_FILE_PATH = os.path.join(CONTEXT_DIR, "SYSTEM.txt")
SYSTEM_EDITOR_PROMPT_FILE_PATH = os.path.join(CONTEXT_DIR, "SYSTEM_EDITOR.txt")
SYSTEM_PROMPT_WITH_FEEDBACK_FILE_PATH = os.path.join(CONTEXT_DIR, "SYSTEM_FEEDBACK.txt")

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


@lru_cache()
def get_faq_questions() -> list[str]:
    """
    Lee las preguntas frecuentes desde el archivo.
    Lanza FileNotFoundError si no existe el archivo.
    """
    if not os.path.exists(FAQ_FILE_PATH):
        raise FileNotFoundError(f"Archivo de FAQ no encontrado: {FAQ_FILE_PATH}")

    with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@lru_cache()
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


@lru_cache()
def get_system_prompt_with_feedback() -> str:
    if not os.path.exists(SYSTEM_PROMPT_WITH_FEEDBACK_FILE_PATH):
        raise FileNotFoundError(
            f"Archivo de prompt del sistema con feedback no encontrado: {SYSTEM_PROMPT_WITH_FEEDBACK_FILE_PATH}"
        )

    with open(SYSTEM_PROMPT_WITH_FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache()
def get_system_editor_prompt() -> str:
    """
    Lee el prompt del sistema desde el archivo.
    Lanza FileNotFoundError si no existe el archivo.
    """
    if not os.path.exists(SYSTEM_EDITOR_PROMPT_FILE_PATH):
        raise FileNotFoundError(
            f"Archivo de prompt del sistema para el editor no encontrado: {SYSTEM_EDITOR_PROMPT_FILE_PATH}"
        )

    with open(SYSTEM_EDITOR_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache()
def get_physical_context() -> str:
    context_files = os.listdir("server/ai/context")
    valid_extensions = (".md", ".txt", ".csv")
    ignored_files = {"SYSTEM.txt", "FAQ.txt", "SYSTEM_EDITOR.txt"}

    context = ""
    for file in context_files:
        if file.endswith(valid_extensions) and file not in ignored_files:
            try:
                with open(f"server/ai/context/{file}", "r", encoding="utf-8") as f:
                    context += f'<FILE name="{file}" used_for="ai_context">\n'
                    context += f.read()
                    context += "</FILE>\n"
            except Exception as e:
                printer.error(f"Error reading file {file}: {e}")
    return context


DEFAULT_WARNING_TEXT = """⚠️ Aviso Importante:

El contenido mostrado, incluyendo: textos, gráficos, imágenes u otro tipo de material incluido en el sitio web denominado ‘Intérprete de Sentencias Judiciales’, tiene exclusivamente una finalidad informativa de lectura simple. Por tanto, no debe ser entendido o concebido como un sustituto de la resolución judicial; en consecuencia, el texto mostrado no tiene ningún valor legal.

Este contenido fue generado automáticamente por inteligencia artificial para facilitar la comprensión general del/los adjunto(s). Puede contener errores u omisiones debido a la calidad del texto, del archivo original o a su complejidad. Esta plataforma no almacena los documentos que usted carga ni guarda información personal o sensible. Su uso es temporal y se elimina una vez concluida la interpretación.
"""


@lru_cache()
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
        # self.check_model(model)
        printer.blue(f"Generating completion using: {model}")
        context_window_size = int(os.getenv("CONTEXT_WINDOW_SIZE", 20000))
        printer.blue(f"Context window size: {context_window_size}")
        response = self.client.chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            options={
                "num_ctx": context_window_size
                # "num_keep": 15,
                # "num_thread": 10,
                # "temperature": 0.8,
            },
        )
        return response.message.content


def cut_user_message(previous_messages: list[dict], n_characters_to_cut: int):
    for message in previous_messages:
        if message["role"] == "user":
            message["content"] = message["content"][:-n_characters_to_cut]
    return previous_messages


class OpenAIProvider:
    def __init__(self, api_key: str, base_url: str = None):
        printer.blue(f"Using OpenAI base URL: {base_url}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def check_model(self, model: str):
        return True

    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        stream: bool = False,
        tools: list[dict] | list[callable] = [],
    ):
        printer.blue(f"Generando respuesta con el modelo: {model}")
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
        )
        RESPONSES_DIR = os.getenv("RESPONSES_DIR", "server/ai/responses")
        # Create the directory if it doesn't exist
        os.makedirs(RESPONSES_DIR, exist_ok=True)

        if DEBUG_MODE:
            random_id = str(uuid.uuid4())
            # Save the response to a file
            with open(f"{RESPONSES_DIR}/{random_id}.json", "w") as f:
                json.dump(response.model_dump(), f)

            # Save the messages to a file
            with open(f"{RESPONSES_DIR}/{random_id}_messages.json", "w") as f:
                json.dump(messages, f)

        if response.choices[0].finish_reason == "length":
            printer.error(
                "El modelo dió una respuesta incompleta. Cortando el mensaje de la última conversación y reintentando."
            )
            messages = cut_user_message(messages, 5000)
            return self.chat(messages, model, stream, tools)

        return response.choices[0].message.content


class AIInterface:
    client: OllamaProvider | OpenAIProvider | None = None

    def __init__(
        self,
        provider: str = "ollama",
        api_key: str = None,
        base_url: str = None,
    ):
        self.provider = provider
        if provider == "ollama":
            self.client = OllamaProvider()
        elif provider == "openai":
            self.client = OpenAIProvider(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Provider {provider} not supported")

        printer.blue("Using AI from", self.provider, "with base URL", base_url)

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

    def check_model(self, model: str):
        return self.client.check_model(model)


def tokenize_prompt(prompt: str):
    # Leer la URL base del .env o usar valor por defecto
    # base_url = os.getenv("PROVIDER_BASE_URL", "http://localhost:8009")
    base_url = "http://localhost:8009"
    url = f"{base_url.rstrip('/')}/tokenize"

    payload = {"prompt": prompt}
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    count = data["count"]
    max_model_len = data["max_model_len"]
    difference = max_model_len - count
    is_difference_more_than_4000 = difference > 4000

    return count, difference, is_difference_more_than_4000
