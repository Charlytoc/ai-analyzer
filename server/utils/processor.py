import os
import hashlib
import json
import time

import re
import uuid
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from server.utils.pdf_reader import DocumentReader
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.ai.ai_interface import (
    AIInterface,
    get_faq_questions,
    get_system_prompt,
    get_system_editor_prompt,
    get_system_prompt_with_feedback,
    get_prompt_from_file,
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


def remove_unwanted_elements(text: str) -> str:
    header_pattern = r"^(#{2,6})\s*(\*\*|__)?\s*¿[^?]+\?\s*(\*\*|__)?\s*$"
    paragraph_pattern = r"^(\*\*|__)?\s*¿[^?]+\?\s*(\*\*|__)?\s*$"
    hr_pattern = r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$"

    text = re.sub(header_pattern, "", text, flags=re.MULTILINE)
    text = re.sub(paragraph_pattern, "", text, flags=re.MULTILINE)
    text = re.sub(hr_pattern, "", text, flags=re.MULTILINE)
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
    # chroma_client = get_chroma_client()
    # number_of_documents = len(document_paths)
    # if number_of_documents > 1:
    #     max_characters_per_document = LIMIT_CHARACTERS_FOR_TEXT // number_of_documents
    # else:
    #     max_characters_per_document = LIMIT_CHARACTERS_FOR_TEXT
    document_reader = DocumentReader()

    complete_text = ""
    # limited_text = ""
    for document_path in document_paths:
        document_text = document_reader.read(document_path)
        printer.green(f"🔍 Documento leído: {document_path}")
        printer.yellow(f"🔍 Inicio del documento: {document_text[:200]}")

        complete_text += f"<document_text name='{document_path}'>: \n{document_text}\n </document_text>"

    if DEBUG_MODE:
        with open("last_complete_text.txt", "w") as f:
            f.write(complete_text)

    return complete_text


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


def read_sources(document_paths: list[str], images_paths: list[str]):
    text_from_all_documents = read_documents(document_paths)
    text_from_all_documents += read_images(images_paths)
    return text_from_all_documents


def ensure_feedback_is_applied(sentence: str):
    printer.blue("🔍 Aplicando retroalimentación a la respuesta...")
    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )
    system_prompt = get_system_prompt_with_feedback()
    if not system_prompt:
        raise ValueError("No se encontró el prompt del sistema.")

    feedback_text = get_feedback_from_redis(n_results=100)
    system_prompt = system_prompt.replace("{{feedback}}", feedback_text)
    system_prompt = system_prompt.replace("{{sentence}}", sentence)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Realiza únicamente las modificaciones necesarias.",
        },
    ]
    response = ai_interface.chat(messages=messages, model=os.getenv("MODEL", "gemma3"))

    _, rejected = was_rejected(response)
    response = clean_reasoning_tag(response)
    response = clean_markdown_block(response)

    if rejected:
        printer.yellow("❌ La respuesta fue rechazada por la IA.")
        response += "\n\n<REJECTED />"

    printer.green("✅ Retroalimentación aplicada a la respuesta.")

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


def generate_sentence_brief(
    source_hash: str,
):
    extracted_data = get_extracted_data(source_hash)
    feedback_text = get_feedback_from_redis(n_results=100)

    system_prompt = get_system_prompt()
    system_prompt = system_prompt.replace("{{feedback}}", feedback_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": extracted_data},
    ]

    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )
    time.sleep(0.5)

    response = ai_interface.chat(messages=messages, model=os.getenv("MODEL", "gemma3"))
    if DEBUG_MODE:
        with open("last_response_before_cleaning.txt", "w") as f:
            f.write(response)

    response, rejected = was_rejected(response)

    response = clean_markdown_block(response)
    response = clean_reasoning_tag(response)
    response = remove_unwanted_elements(response)
    if not is_spanish(response[:150]):
        printer.yellow("🔍 La respuesta no está en español, traduciendo...")
        printer.yellow(f"🔍 Respuesta original: {response}")
        response = translate_to_spanish(response)
        response = clean_markdown_block(response)
        response = clean_reasoning_tag(response)
        response = remove_unwanted_elements(response)
    else:
        printer.green("🔍 La respuesta ya está en español en el primer intento.")
    # response = ensure_feedback_is_applied(response)

    redis_cache.set(
        f"sentence_brief:{source_hash}",
        json.dumps(
            {
                "sentence": response,
                "message": (
                    "Sentencia ciudadana generada con éxito."
                    if not rejected
                    else "Sentencia ciudadana rechazada."
                ),
                "workflow": "update" if not rejected else "rejected",
                "rejected": rejected,
            }
        ),
        ex=EXPIRATION_TIME,
    )
    printer.green(f"💾 Sentencia ciudadana guardada en cache: {source_hash}")

    return response


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


class UpdateResponse(BaseModel):
    workflow: Literal["update", "question", "rejected"] = Field(
        ...,
        description="Indica el tipo de interacción que se debe realizar con el usuario.",
    )
    rejected: bool = Field(
        ...,
        description="Indica explícitamente si la solicitud de cambios fue rechazada o no.",
    )
    message: str = Field(
        ...,
        description="Mensaje de la respuesta para el usuario, si la solicitud de cambios fue rechazada, retorna un mensaje amigable indicando por qué debes rechazar la solicitud y la forma correcta de poder solicitar cambios. Si la solicitud de cambios fue aceptada, retorna un mensaje amigable indicando que se realizaron los cambios y que si quiere hacer más cambios, puede hacerlo.",
    )
    sentence: str = Field(
        ...,
        description="La respuesta final con los cambios realizados únicamente si el workflow es 'update', si no se realizaron cambios, retorna 'unchanged'.",
    )


def update_sentence_brief(
    sources_hash: str, sentence: str, changes: str, prev_messages: str
):
    system_editor_prompt = get_system_editor_prompt()
    messages = [
        {
            "role": "system",
            "content": system_editor_prompt.replace("{{sentencia}}", sentence).replace(
                "{{prev_messages}}", prev_messages
            ),
        },
        {
            "role": "user",
            "content": changes,
        },
    ]

    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )
    printer.yellow("🔍 Enviando mensaje al editor...")
    response = ai_interface.chat_structured(
        messages=messages,
        model=os.getenv("MODEL", "gemma3"),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "update-sentence-schema",
                "schema": UpdateResponse.model_json_schema(),
            },
        },
    )

    printer.yellow("🔍 Respuesta del editor: ", response.content)
    update_response = UpdateResponse.model_validate_json(response.content)

    if DEBUG_MODE:
        with open("last_update_response.txt", "w") as f:
            f.write(update_response.model_dump_json())

    # clean the response
    update_response.sentence = clean_reasoning_tag(update_response.sentence)
    update_response.sentence = clean_markdown_block(update_response.sentence)
    update_response.sentence = remove_unwanted_elements(update_response.sentence)
    update_response.sentence = update_response.sentence.strip()

    redis_cache.set(
        f"sentence_brief:{sources_hash}",
        update_response.model_dump_json(),
        ex=EXPIRATION_TIME,
    )
    return update_response


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


def get_source_text(source_hash: str):
    source_text = redis_cache.get(f"source_text:{source_hash}")
    if not source_text:
        raise Exception("No se encontró el texto de origen en Redis")
    return source_text


def get_extracted_data(source_hash: str):
    extracted_data = redis_cache.get(f"extracted_data:{source_hash}")
    if not extracted_data:
        raise Exception("No se encontró el texto de origen en Redis")
    return extracted_data


def split_text_in_chunks(text: str, n_characters: int) -> list[str]:
    """
    Splits the input text into chunks of at most n_characters each.

    Args:
        text (str): The text to split.
        n_characters (int): The maximum number of characters per chunk.

    Returns:
        list[str]: List of text chunks.
    """
    return [text[i : i + n_characters] for i in range(0, len(text), n_characters)]


def extract_data_from_chunk(chunk: str):
    system_prompt = get_prompt_from_file("EXTRACTOR")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk},
    ]
    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )
    response = ai_interface.chat(messages=messages, model=os.getenv("MODEL", "gemma3"))
    return response


def sequencial_extraction(source_hash: str) -> str:
    source_text = get_source_text(source_hash)
    chunks = split_text_in_chunks(source_text, n_characters=40000)
    printer.yellow(f"🔍 N chunks: {len(chunks)}")

    cummulative_response = ""

    for i, chunk in enumerate(chunks):
        response = extract_data_from_chunk(chunk)
        printer.yellow(f"🔍 Respuesta del chunk {i}...: {response}")
        cummulative_response += response + "\n"

    redis_cache.set(
        f"extracted_data:{source_hash}", cummulative_response, ex=EXPIRATION_TIME
    )

    return cummulative_response


def generate_feedback_from_messages(sources_hash: str, messages: str):
    system_prompt = get_prompt_from_file("FEEDBACK_GENERATOR")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": messages},
    ]
    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )
    res = ai_interface.chat(messages=messages, model=os.getenv("MODEL", "gemma3"))
    printer.yellow("🔍 Feedback generado: ", res)
    redis_cache.set(
        f"feedback:{sources_hash}",
        res,
        ex=EXPIRATION_TIME,
    )
    return res
