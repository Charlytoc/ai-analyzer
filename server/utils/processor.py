import os
import hashlib
import json

from typing import Literal
from pydantic import BaseModel, Field, field_validator
from server.utils.pdf_reader import DocumentReader
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.utils.ai_interface import AIInterface, get_physical_context
from server.utils.image_reader import ImageReader
from server.ai.vector_store import chroma_client
from server.utils.detectors import is_spanish

EXPIRATION_TIME = 60 * 60 * 24 * 30  # 30 days

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", None)
LIMIT_CHARACTERS_FOR_TEXT = 100000

CONTEXT_DIR = os.getenv("CONTEXT_DIR", "server/ai/context")
FAQ_FILE_PATH = os.path.join(CONTEXT_DIR, "FAQ.txt")
SYSTEM_PROMPT_FILE_PATH = os.path.join(CONTEXT_DIR, "SYSTEM.txt")

printer = Printer("ROUTES")
redis_cache = RedisCache()


class DataSource(BaseModel):
    type: Literal["document", "image"]
    name: str
    content: str
    hash: str = Field(default="")

    @field_validator("hash", mode="after", check_fields=False)
    def compute_hash(cls, v, info):
        txt = info.data.get("content", "")
        return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def hasher(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def flatten_list(nested_list):
    """Aplana una lista de listas en una sola lista de elementos."""
    if not nested_list:
        return []
    return [item for sublist in nested_list for item in sublist]


def get_faq_results(doc_hash: str):
    results_str = ""

    questions = get_faq_questions()
    # printer.green(f"Preguntas para extraer información del documento: {questions}")

    documents = set()
    for question in questions:
        retrieval = chroma_client.get_results(
            collection_name=f"doc_{doc_hash}",
            query_texts=[question],
            n_results=3,
        )

        _documents = flatten_list(retrieval.get("documents", []))
        documents.update(_documents)

    results_str += (
        f"Lista de preguntas para la base de datos vectorial: {' '.join(questions)}"
    )
    results_str += f"Resultados de la búsqueda: {' '.join(documents)}"
    return results_str


DEFAULT_WARNING_TEXT = """⚠ Aviso Importante:

El contenido del documento mostrado, como pueden ser los textos, los gráficos, las imágenes y otro tipo de material incluido en el Sitio Web de "Sentencia Ciudadana", tiene exclusivamente una finalidad informativa de lectura simple. El contenido no se ha concebido como sustituto de la resolución judicial, en consecuencia, el texto mostrado no tiene valor legal.

Este resumen fue generado automáticamente por inteligencia artificial para facilitar la comprensión general del/los adjunto(s). Puede contener errores u omisiones debido a la calidad del texto, del archivo original o a su complejidad.
"""


def get_warning_text():
    warning_text = os.getenv("WARNING_TEXT", DEFAULT_WARNING_TEXT)
    if not warning_text:
        return DEFAULT_WARNING_TEXT
    return warning_text


def translate_to_spanish(text: str):
    system_prompt = """
    Your task is to translate the given text to spanish, preserve the original meaning and structure of the text. Return only the translated text, without any other text or explanation. Your unique response must be the translated text.
    """
    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"), api_key=os.getenv("OLLAMA_API_KEY")
    )
    response = ai_interface.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=os.getenv("MODEL", "gemma3"),
    )
    return response


def generate_sentence_brief(
    document_paths: list[str], images_paths: list[str], extra: dict = {}
):
    physical_context = get_physical_context()

    use_cache = extra.get("use_cache", True)

    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"), api_key=os.getenv("OLLAMA_API_KEY")
    )

    system_prompt = get_system_prompt()
    if not system_prompt:
        raise ValueError("No se encontró el prompt del sistema.")

    printer.blue("Usando System Prompt:")
    printer.yellow(system_prompt)

    if physical_context:
        system_prompt = system_prompt.replace("{{context}}", physical_context)
    if len(get_faq_questions()) > 0:
        system_prompt = system_prompt.replace("{{faq}}", "\n".join(get_faq_questions()))

    messages = [{"role": "system", "content": system_prompt}]
    document_reader = DocumentReader()

    number_of_documents = len(document_paths)
    if number_of_documents > 1:
        max_characters_per_document = LIMIT_CHARACTERS_FOR_TEXT // number_of_documents
    else:
        max_characters_per_document = LIMIT_CHARACTERS_FOR_TEXT

    text_from_all_documents = ""
    for document_path in document_paths:
        document_text = document_reader.read(document_path)
        printer.green(f"🔍 Documento leído: {document_path}")
        printer.yellow(f"🔍 Inicio del documento: {document_text[:200]}")

        with open(
            f"uploads/documents/read/{os.path.basename(document_path)}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(document_text)

        document_hash = hasher(document_text)

        truncated = document_text[:max_characters_per_document]

        if len(truncated) == len(document_text):
            printer.yellow(f"🔍 Se agrega todo el documento: {document_path}")
            text_from_all_documents += f"<document_text name='{document_path}'>: \n{document_text}\n </document_text>"
        else:
            printer.yellow(
                f"🔍 Se agrega parte del documento y el resto es vectorizado: {document_path}"
            )
            text_from_all_documents += f"<document_text name='{document_path}'>: \n{truncated}\n </document_text>"
            created = chroma_client.get_collection_or_none(f"doc_{document_hash}")
            if not created:
                chroma_client.get_or_create_collection(f"doc_{document_hash}")
                chunks = chroma_client.chunkify(
                    document_text, chunk_size=1000, chunk_overlap=200
                )
                chroma_client.bulk_upsert_chunks(
                    collection_name=f"doc_{document_hash}",
                    chunks=chunks,
                )

            faq_results = get_faq_results(document_hash)
            text_from_all_documents += f"<faq_results for_document='{document_path}'>: {faq_results}</faq_results>"

    for image_path in images_paths:
        image_reader = ImageReader()
        image_text = image_reader.read(image_path)
        printer.yellow(f"🔍 Imagen leída: {image_path}")
        printer.yellow(f"🔍 Inicio de la imagen: {image_text[:200]}")
        text_from_all_documents += (
            f"<image_text name={image_path}>: {image_text} </image_text>"
        )

    messages.append(
        {
            "role": "user",
            "content": f"# TEXT FROM ALL SOURCES\n\n{text_from_all_documents}",
        }
    )

    messages_json = json.dumps(messages, sort_keys=True, indent=4)
    # Save the messages to a file
    with open("messages.json", "w", encoding="utf-8") as f:
        f.write(messages_json)
    messages_hash = hashlib.sha256(messages_json.encode("utf-8")).hexdigest()

    if use_cache:
        cached_response = redis_cache.get(messages_hash)
        if cached_response:
            printer.green(f"👀 Sentencia ciudadana cacheada: {messages_hash}")
            return cached_response, True

    printer.red(f"🔍 No se encontró la sentencia ciudadana en cache: {messages_hash}")
    response = ai_interface.chat(messages=messages, model=os.getenv("MODEL", "gemma3"))
    if not is_spanish(response[:150]):
        printer.red("🔍 La respuesta no está en español, traduciendo...")
        response = translate_to_spanish(response)
    else:
        printer.green("🔍 La respuesta ya está en español en el primer intento.")
    response = response + "\n\n" + get_warning_text()

    redis_cache.set(messages_hash, response, ex=EXPIRATION_TIME)
    printer.green(f"💾 Sentencia ciudadana guardada en cache: {messages_hash}")

    return response, False
