import shutil
import subprocess
import requests
from ollama import Client
import os
from .printer import Printer
from openai import OpenAI

printer = Printer("AI INTERFACE")


def get_physical_context() -> str:
    # Reads all the files from server/ai/context
    context_files = os.listdir("server/ai/context")
    context_files = [
        f for f in context_files if f.endswith(".md") or f.endswith(".txt")
    ]
    ignored_files = ["SYSTEM.txt", "FAQ.txt"]
    context_files = [f for f in context_files if f not in ignored_files]
    context = ""
    for file in context_files:
        with open(f"server/ai/context/{file}", "r", encoding="utf-8") as f:
            context += f'<FILE name="{file}" used_for="ai_context">\n'
            context += f.read()
            context += "</FILE>"
    return context


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
            options={"num_ctx": 100000},
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
