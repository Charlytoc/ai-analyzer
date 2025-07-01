import os
import hashlib

import re
import uuid
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from server.utils.pdf_reader import DocumentReader
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.ai.ai_interface import (
    AIInterface,
    # get_physical_context,
    get_faq_questions,
    get_system_prompt,
    get_system_editor_prompt,
    get_warning_text,
)
from fastapi import UploadFile
from typing import List, Tuple

from server.utils.image_reader import ImageReader
from server.ai.vector_store import get_chroma_client
from server.utils.detectors import is_spanish


EXPIRATION_TIME = 60 * 60 * 24  # 24 horas
LIMIT_CHARACTERS_FOR_TEXT = 10000

N_CHARACTERS_FOR_FEEDBACK_VECTORIZATION = 3000

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


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


def flatten_list(nested_list):
    """Aplana una lista de listas en una sola lista de elementos."""
    if not nested_list:
        return []
    return [item for sublist in nested_list for item in sublist]


def remove_duplicates(lst):
    """Elimina duplicados de una lista manteniendo el orden."""
    seen = set()
    return [item for item in lst if not (item in seen or seen.add(item))]


def get_faq_results(doc_hash: str):
    chroma_client = get_chroma_client()
    results_str = ""

    questions = get_faq_questions()

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
        f"Número de preguntas para la base de datos vectorial: {len(questions)}"
    )
    documents = remove_duplicates(documents)
    results_str += (
        f"Resultados de la búsqueda en base de datos vectorial: {' '.join(documents)}"
    )
    # Save as a file called "faq_results.txt"
    with open("faq_results.txt", "w") as f:
        f.write(results_str)

    try:
        chroma_client.delete_collection(f"doc_{doc_hash}")
    except Exception as e:
        printer.error(f"❌ Error al eliminar la colección en vector store: {e}")
    return results_str


def translate_to_spanish(text: str):
    system_prompt = """
    Your task is to translate the given text to spanish, preserve the original meaning and structure of the text. Return only the translated text, without any other text or explanation. Your unique response must be the translated text.
    """
    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )
    response = ai_interface.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=os.getenv("MODEL", "gemma3"),
    )
    return response


def clean_markdown_block(text: str) -> str:
    start_tag = "```markdown"
    end_tag = "```"

    start_index = text.find(start_tag)
    if start_index == -1:
        return text

    start_index += len(start_tag)
    end_index = text.find(end_tag, start_index)
    if end_index == -1:
        return text

    printer.yellow("🔍 La respusta está dentro de un bloque markdown, limpiando...")
    content = text[start_index:end_index]
    return content.strip()


def clean_reasoning_tag(text: str):
    # Print the reasoning content
    end_index = text.find("</think>")
    if end_index == -1:
        return text
    return text[end_index + len("</think>") :].lstrip()


def remove_h2_h6_questions_and_paragraph_questions(text: str) -> str:
    header_pattern = r"^(#{2,6})\s*(\*\*|__)?\s*¿[^?]+\?\s*(\*\*|__)?\s*$"
    paragraph_pattern = r"^(\*\*|__)?\s*¿[^?]+\?\s*(\*\*|__)?\s*$"

    text = re.sub(header_pattern, "", text, flags=re.MULTILINE)
    text = re.sub(paragraph_pattern, "", text, flags=re.MULTILINE)
    return text.strip()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".md"}


def get_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def validate_attachments(
    images: List[UploadFile], documents: List[UploadFile]
) -> Tuple[List[UploadFile], List[UploadFile]]:
    """
    Clasifica y valida archivos recibidos como imágenes y documentos.
    Devuelve dos listas: (imagenes_validas, documentos_validos).
    Lanza HTTPException si encuentra una extensión no permitida.
    """
    valid_images = []
    valid_documents = []

    # Procesa archivos enviados como imágenes
    for file in images:
        ext = get_extension(file.filename)
        if ext in IMAGE_EXTENSIONS:
            valid_images.append(file)
        elif ext in DOCUMENT_EXTENSIONS:
            valid_documents.append(file)
        else:
            printer.error(
                f"❌ Extensión no permitida en archivo, archivo ignorado: {file.filename}"
            )
            # raise HTTPException(
            #     status_code=400,
            #     detail=f"Extensión no permitida en archivo: {file.filename}"
            # )

    # Procesa archivos enviados como documentos
    for file in documents:
        ext = get_extension(file.filename)
        if ext in DOCUMENT_EXTENSIONS:
            valid_documents.append(file)
        elif ext in IMAGE_EXTENSIONS:
            valid_images.append(file)
        else:
            printer.error(
                f"❌ Extensión no permitida en archivo, archivo ignorado: {file.filename}"
            )
            # raise HTTPException(
            #     status_code=400,
            #     detail=f"Extensión no permitida en archivo: {file.filename}"
            # )

    return valid_images, valid_documents


def read_documents(document_paths: list[str]):
    chroma_client = get_chroma_client()
    number_of_documents = len(document_paths)
    if number_of_documents > 1:
        max_characters_per_document = LIMIT_CHARACTERS_FOR_TEXT // number_of_documents
    else:
        max_characters_per_document = LIMIT_CHARACTERS_FOR_TEXT
    document_reader = DocumentReader()

    complete_text = ""
    limited_text = ""
    for document_path in document_paths:
        document_text = document_reader.read(document_path)
        printer.green(f"🔍 Documento leído: {document_path}")
        printer.yellow(f"🔍 Inicio del documento: {document_text[:200]}")

        document_hash = hasher(document_text)

        truncated = document_text[:max_characters_per_document]

        complete_text += f"<document_text name='{document_path}'>: \n{document_text}\n </document_text>"

        if len(truncated) == len(document_text):
            printer.yellow(f"🔍 Se agrega todo el documento: {document_path}")
            limited_text += f"<document_text name='{document_path}'>: \n{document_text}\n </document_text>"
        else:
            printer.yellow(
                f"🔍 Se agrega parte del documento y el resto es vectorizado: {document_path}"
            )
            limited_text += f"<document_text name='{document_path}'>: \n{truncated}\n </document_text>"
            printer.yellow(f"🔍 Caracteres antes de vectorizar: {len(limited_text)}")
            created = chroma_client.get_collection_or_none(f"doc_{document_hash}")
            if not created:
                printer.blue(
                    "🔍 Creando colección en vector store para el documento..."
                )
                chroma_client.get_or_create_collection(f"doc_{document_hash}")
                chunks = chroma_client.chunkify(
                    document_text, chunk_size=1500, chunk_overlap=400
                )
                chroma_client.bulk_upsert_chunks(
                    collection_name=f"doc_{document_hash}",
                    chunks=chunks,
                )

            faq_results = get_faq_results(document_hash)
            printer.yellow(f"🔍 FAQ results length: {len(faq_results)}")
            limited_text += f"<faq_results for_document='{document_path}'>: {faq_results}</faq_results>"
            printer.yellow(f"🔍 Caracteres después de vectorizar: {len(limited_text)}")

    if DEBUG_MODE:
        with open("last_complete_text.txt", "w") as f:
            f.write(complete_text)

    return limited_text, complete_text


def read_images(images_paths: list[str]):
    image_reader = ImageReader()
    text_from_all_documents = ""
    for image_path in images_paths:
        image_text = image_reader.read(image_path)
        printer.yellow(f"🔍 Imagen leída: {image_path}")
        printer.yellow(f"🔍 Inicio de la imagen: {image_text[:200]}")
        text_from_all_documents += (
            f"<image_text name={image_path}>: {image_text} </image_text>"
        )
    return text_from_all_documents


def format_messages(document_paths: list[str], images_paths: list[str]):

    system_prompt = get_system_prompt()
    if not system_prompt:
        raise ValueError("No se encontró el prompt del sistema.")

    if len(get_faq_questions()) > 0:
        system_prompt = system_prompt.replace("{{faq}}", "\n".join(get_faq_questions()))

    text_from_all_documents, complete_text = read_documents(document_paths)
    text_from_all_documents += read_images(images_paths)

    user_message_text = f"# Estas son las fuentes de información disponibles para escribir la sentencia:\n\n{text_from_all_documents}"

    feedback_text = get_feedback_from_redis(n_results=50)

    if feedback_text:
        system_prompt = system_prompt.replace(
            "{{feedback}}",
            f"{feedback_text}",
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages.append(
        {
            "role": "user",
            "content": user_message_text,
        }
    )

    return messages, complete_text


def generate_sentence_brief(
    messages: list[dict],
    sentence_hash: str,
):

    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )

    response = ai_interface.chat(messages=messages, model=os.getenv("MODEL", "gemma3"))
    if DEBUG_MODE:
        with open("last_response_before_cleaning.txt", "w") as f:
            f.write(response)
            
    response = clean_markdown_block(response)
    response = clean_reasoning_tag(response)
    response = remove_h2_h6_questions_and_paragraph_questions(response)
    if not is_spanish(response[:150]):
        printer.yellow("🔍 La respuesta no está en español, traduciendo...")
        printer.yellow(f"🔍 Respuesta original: {response}")
        response = translate_to_spanish(response)
        response = clean_markdown_block(response)
        response = clean_reasoning_tag(response)
        response = remove_h2_h6_questions_and_paragraph_questions(response)
    else:
        printer.green("🔍 La respuesta ya está en español en el primer intento.")

    redis_cache.set(f"sentence_brief:{sentence_hash}", response, ex=EXPIRATION_TIME)
    printer.green(f"💾 Sentencia ciudadana guardada en cache: {sentence_hash}")

    return response


def was_rejected(response: str) -> tuple[str, bool]:
    """
    Busca si la respuesta contiene alguna de las etiquetas de rechazo.
    Si existe, las elimina y retorna el texto limpio y True.
    Si no, retorna el texto original y False.
    """
    tags = ["<REJECTED />", "<rejected />", "<rechazado />"]
    found = any(tag in response for tag in tags)
    if found:
        for tag in tags:
            response = response.replace(tag, "")
    return response, found


def update_system_prompt(previous_messages: list[dict], new_system_prompt: str):

    for message in previous_messages:
        if message["role"] == "system":
            message["content"] = new_system_prompt

    previous_messages = previous_messages[:2]
    return previous_messages


def change_user_message(previous_messages: list[dict], new_user_message: str):
    for message in previous_messages:
        if message["role"] == "user":
            message["content"] = new_user_message
    return previous_messages


def update_sentence_brief(hash: str, sentence: str, changes: str):
    # previous_messages = redis_cache.get(f"messages_input:{hash}")
    # previous_messages = json.loads(previous_messages)
    system_editor_prompt = get_system_editor_prompt()
    messages = [
        {
            "role": "system",
            "content": system_editor_prompt.replace("{{sentencia}}", sentence),
        },
        {
            "role": "user",
            "content": f"-----\nPor favor realiza únicamente los cambios que se te indican a continuación. Debes retornar únicamente el texto correspondiente a la sentencia ciudadana con los cambios realizados. Los cambios que debes realizar son: {changes}.",
        },
    ]

    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )
    response = ai_interface.chat(
        messages=messages,
        model=os.getenv("MODEL", "gemma3"),
    )

    if DEBUG_MODE:
        with open("last_editor_response_before_cleaning.txt", "w") as f:
            f.write(response)

    response = clean_reasoning_tag(response)
    _, rejected = was_rejected(response)
    if rejected:
        printer.yellow("❌ La respuesta fue rechazada por la IA.")

    response = clean_markdown_block(response)

    printer.green(f"✅ Respuesta final al reescribir la sentencia: {response}")
    redis_cache.set(f"sentence_brief:{hash}", response, ex=EXPIRATION_TIME)
    return response


def generate_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def generate_random_id():
    return str(uuid.uuid4())


def upsert_feedback_in_redis(feedback: str):
    key = "all_feedbacks"
    try:
        redis_cache.rpush(key, feedback)
        printer.green(f"💾 Feedback guardado en Redis: {feedback}")
        return True
    except Exception as e:
        printer.error(f"❌ Error al guardar el feedback en Redis: {e}")
        return False


def get_feedback_from_redis(n_results: int = 10):
    key = "all_feedbacks"
    try:
        printer.blue("🔍 Buscando feedbacks en Redis...")
        # Obtener los últimos n_results feedbacks
        feedbacks = redis_cache.lrange(key, -n_results, -1)
        # Redis devuelve bytes, decodifica si es necesario
        feedbacks = [
            fb.decode("utf-8") if isinstance(fb, bytes) else fb for fb in feedbacks
        ]
        printer.green(f"🔍 Feedbacks encontrados: {len(feedbacks)} feedbacks")
        return "\n".join(feedbacks)
    except Exception as e:
        printer.error(f"❌ Error al obtener feedbacks de Redis: {e}")
        return ""


def format_response(
    response: str, cached: bool, hash: str, n_documents: int, n_images: int
):
    return {
        "status": "SUCCESS",
        "message": "Sentencia ciudadana generada con éxito.",
        "brief": response,
        "n_documents": n_documents,
        "n_images": n_images,
        "cache_used": cached,
        "hash": hash,
        "warning": get_warning_text(),
    }
