import os
import hashlib
import json
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from server.utils.pdf_reader import DocumentReader
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.utils.ai_interface import (
    AIInterface,
    get_physical_context,
    get_faq_questions,
    get_system_prompt,
    get_warning_text
)
from server.utils.image_reader import ImageReader
from server.ai.vector_store import chroma_client
from server.utils.detectors import is_spanish

EXPIRATION_TIME = 60 * 60 * 24 * 30
LIMIT_CHARACTERS_FOR_TEXT = 100000


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
    documents = remove_duplicates(documents)
    results_str += f"Resultados de la búsqueda: {' '.join(documents)}"
    return results_str


def translate_to_spanish(text: str):
    system_prompt = """
    Your task is to translate the given text to spanish, preserve the original meaning and structure of the text. Return only the translated text, without any other text or explanation. Your unique response must be the translated text.
    """
    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
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
    stripped = text.strip()

    if not stripped.startswith("```markdown"):
        return text

    # Remove the opening line
    content = stripped[len("```markdown") :].lstrip()

    # Find the last occurrence of closing ```
    closing_index = content.rfind("```")
    if closing_index == -1:
        return text  # No proper closing block

    return content[:closing_index].rstrip()


def generate_sentence_brief(
    document_paths: list[str], images_paths: list[str], extra: dict = {}
):
    physical_context = get_physical_context()

    printer.blue("Usando Contexto Físico:")
    printer.yellow(physical_context)

    use_cache = extra.get("use_cache", True)

    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
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

    user_message_text = f"# TEXT FROM ALL SOURCES\n\n{text_from_all_documents}"

    messages.append(
        {
            "role": "user",
            "content": user_message_text,
        }
    )

    messages_json = json.dumps(messages, sort_keys=True, indent=4)
    messages_hash = hashlib.sha256(messages_json.encode("utf-8")).hexdigest()

    feedback_text = get_feedback_from_vector_store(user_message_text)
    if feedback_text:
        messages.append(
            {
                "role": "user",
                "content": f"# You previously received this feedback on other sentences, try to not repeat yourself and avoid the same mistakes:\n\n{feedback_text}",
            }
        )
    redis_cache.set(
        f"messages_input:{messages_hash}", messages_json, ex=EXPIRATION_TIME
    )

    if use_cache:
        cached_response = redis_cache.get(f"sentence_brief:{messages_hash}")
        if cached_response:
            printer.green(f"👀 Sentencia ciudadana cacheada: {messages_hash}")
            return cached_response, True, messages_hash

    printer.red(f"🔍 No se encontró la sentencia ciudadana en cache: {messages_hash}")
    response = ai_interface.chat(messages=messages, model=os.getenv("MODEL", "gemma3"))
    response = clean_markdown_block(response)
    if not is_spanish(response[:150]):
        printer.red("🔍 La respuesta no está en español, traduciendo...")
        response = translate_to_spanish(response)
        response = clean_markdown_block(response)
    else:
        printer.green("🔍 La respuesta ya está en español en el primer intento.")

    response = response + "\n\n" + get_warning_text()
    redis_cache.set(f"sentence_brief:{messages_hash}", response, ex=EXPIRATION_TIME)
    printer.green(f"💾 Sentencia ciudadana guardada en cache: {messages_hash}")

    return response, False, messages_hash


def update_sentence_brief(hash: str, changes: str):
    sentence = redis_cache.get(f"sentence_brief:{hash}")
    previous_messages = redis_cache.get(f"messages_input:{hash}")
    previous_messages = json.loads(previous_messages)
    previous_messages.append(
        {
            "role": "user",
            "content": f"Por favor realiza cambios, no estoy conforme con el resultado. Los cambios que debes realizar son: {changes}",
        }
    )
    if not sentence:
        raise ValueError("No se encontró la sentencia ciudadana.")

    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
    )
    response = ai_interface.chat(
        messages=previous_messages,
        model=os.getenv("MODEL", "gemma3"),
    )

    return response


def generate_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_user_message_partial_text(messages: list[dict]):
    for message in messages:
        if message["role"] == "user":
            return message["content"][:1000]
    return ""


def upsert_feedback_in_vector_store(hash: str, feedback: str):
    try:
        previous_messages = redis_cache.get(f"messages_input:{hash}")
        previous_messages = json.loads(previous_messages)
        partial_text = get_user_message_partial_text(previous_messages)

        if not partial_text:
            raise ValueError("No se encontró el mensaje del usuario.")

        chroma_client.upsert_chunk(
            collection_name="sentence_feedbacks",
            chunk_text=partial_text,
            chunk_id=f"feedback_{generate_id(partial_text)}",
            metadata={"feedback": feedback},
        )

        printer.green(f"💾 Feedback guardado en vector store: {feedback}")

        return True
    except Exception as e:
        printer.red(f"❌ Error al guardar el feedback en el vector store: {e}")
        return False


def get_feedback_from_vector_store(documents_text: str):
    trimmed_text = documents_text[:1000]
    try:
        printer.blue("🔍 Buscando feedback en vector store...")
        chunks = chroma_client.get_results(
            collection_name="sentence_feedbacks",
            query_texts=[trimmed_text],
            n_results=5,
        )

        feedbacks = []
        for i in range(len(chunks["metadatas"])):
            feedback = chunks["metadatas"][i][0]["feedback"]
            feedbacks.append(feedback)

        printer.green(f"🔍 Feedback encontrado: {feedbacks}")
        return "\n".join(feedbacks)
    except Exception as e:
        printer.red(f"❌ Error al obtener el feedback del vector store: {e}")
        return ""
