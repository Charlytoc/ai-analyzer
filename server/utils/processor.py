import os
import hashlib
import json
import uuid
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from server.utils.pdf_reader import DocumentReader
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.ai.ai_interface import (
    AIInterface,
    get_physical_context,
    get_faq_questions,
    get_system_prompt,
    get_system_editor_prompt,
    get_warning_text,
)
from server.utils.image_reader import ImageReader
from server.ai.vector_store import get_chroma_client
from server.utils.detectors import is_spanish

EXPIRATION_TIME = 60 * 60 * 24 * 30
LIMIT_CHARACTERS_FOR_TEXT = int(os.getenv("LIMIT_CHARACTERS_FOR_ANALYSIS", 40000))

N_CHARACTERS_FOR_FEEDBACK_VECTORIZATION = 3000

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


def read_documents(document_paths: list[str]):
    chroma_client = get_chroma_client()
    number_of_documents = len(document_paths)
    if number_of_documents > 1:
        max_characters_per_document = LIMIT_CHARACTERS_FOR_TEXT // number_of_documents
    else:
        max_characters_per_document = LIMIT_CHARACTERS_FOR_TEXT
    document_reader = DocumentReader()

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

    return text_from_all_documents


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
    physical_context = get_physical_context()

    system_prompt = get_system_prompt()
    if not system_prompt:
        raise ValueError("No se encontró el prompt del sistema.")

    if physical_context:
        system_prompt = system_prompt.replace("{{context}}", physical_context)
    if len(get_faq_questions()) > 0:
        system_prompt = system_prompt.replace("{{faq}}", "\n".join(get_faq_questions()))

    text_from_all_documents = read_documents(document_paths)
    text_from_all_documents += read_images(images_paths)

    user_message_text = f"# These are the sources of information to write the sentence:\n\n{text_from_all_documents}"

    feedback_text = get_feedback_from_vector_store(user_message_text)

    if feedback_text:
        system_prompt += f"---FEEDBACK---\n\nYou previously received this feedback on other sentences, try to not repeat yourself and avoid the same mistakes:\n\n{feedback_text}\n\n---END_OF_FEEDBACK---\n\n"

    messages = [{"role": "system", "content": system_prompt}]
    # messages = []
    messages.append(
        # {
        #     "role": "user",
        #     "content": "<SYSTEM>\n\n"
        #     + system_prompt
        #     + "\n\n</SYSTEM>\n\n<USER>\n\n"
        #     + user_message_text
        #     + "\n\n</USER>",
        # }
        {
            "role": "user",
            "content": user_message_text,
        }
    )

    return messages


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
    response = clean_markdown_block(response)
    if not is_spanish(response[:150]):
        printer.yellow("🔍 La respuesta no está en español, traduciendo...")
        response = translate_to_spanish(response)
        response = clean_markdown_block(response)
    else:
        printer.green("🔍 La respuesta ya está en español en el primer intento.")

    response = response
    redis_cache.set(f"sentence_brief:{sentence_hash}", response, ex=EXPIRATION_TIME)
    printer.green(f"💾 Sentencia ciudadana guardada en cache: {sentence_hash}")

    return response


def update_system_prompt(previous_messages: list[dict], new_system_prompt: str):

    for message in previous_messages:
        if message["role"] == "system":
            message["content"] = new_system_prompt

    previous_messages = previous_messages[:2]
    return previous_messages


def append_to_user_message(previous_messages: list[dict], new_user_message: str):
    for message in previous_messages:
        if message["role"] == "user":
            message["content"] = message["content"] + "\n\n" + new_user_message
    return previous_messages


def update_sentence_brief(hash: str, changes: str):
    sentence = redis_cache.get(f"sentence_brief:{hash}")
    previous_messages = redis_cache.get(f"messages_input:{hash}")
    previous_messages = json.loads(previous_messages)
    system_editor_prompt = get_system_editor_prompt()
    previous_messages = update_system_prompt(
        previous_messages,
        system_editor_prompt.replace("{{sentencia}}", sentence),
    )
    previous_messages = append_to_user_message(
        previous_messages,
        f"-----\nPor favor realiza únicamente los cambios que se te indican a continuación. Debes retornar únicamente el texto correspondiente a la sentencia ciudadana con los cambios realizados. Los cambios que debes realizar son: {changes}",
    )
    if not sentence:
        raise ValueError("No se encontró la sentencia ciudadana.")

    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"),
        api_key=os.getenv("PROVIDER_API_KEY", "asdasd"),
        base_url=os.getenv("PROVIDER_BASE_URL", None),
    )
    response = ai_interface.chat(
        messages=previous_messages,
        model=os.getenv("MODEL", "gemma3"),
    )
    response = clean_markdown_block(response)
    redis_cache.set(f"sentence_brief:{hash}", response, ex=EXPIRATION_TIME)
    return response


def generate_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def generate_random_id():
    return str(uuid.uuid4())


def get_user_message_partial_text(messages: list[dict]):
    for message in messages:
        if message["role"] == "user":
            return message["content"][:N_CHARACTERS_FOR_FEEDBACK_VECTORIZATION]
    return ""


def upsert_feedback_in_vector_store(hash: str, feedback: str):
    chroma_client = get_chroma_client()
    try:
        previous_messages = redis_cache.get(f"messages_input:{hash}")
        previous_messages = json.loads(previous_messages)
        partial_text = get_user_message_partial_text(previous_messages)

        if not partial_text:
            raise ValueError("No se encontró el mensaje del usuario.")

        chroma_client.upsert_chunk(
            collection_name="sentence_feedbacks",
            chunk_text=partial_text,
            chunk_id=f"feedback_{generate_random_id()}",
            metadata={"feedback": feedback},
        )

        printer.green(f"💾 Feedback guardado en vector store: {feedback}")

        return True
    except Exception as e:
        printer.error(f"❌ Error al guardar el feedback en el vector store: {e}")
        return False


def get_feedback_from_vector_store(documents_text: str):
    chroma_client = get_chroma_client()
    trimmed_text = documents_text[:N_CHARACTERS_FOR_FEEDBACK_VECTORIZATION]
    try:
        printer.blue("🔍 Buscando feedback en vector store...")
        chunks = chroma_client.get_results(
            collection_name="sentence_feedbacks",
            query_texts=[trimmed_text],
            n_results=10,
        )

        feedbacks = []
        for i in range(len(chunks["metadatas"])):
            feedback = chunks["metadatas"][i][0]["feedback"]
            feedbacks.append(feedback)

        printer.green(f"🔍 Feedback encontrado: {feedbacks}")
        return "\n".join(feedbacks)
    except Exception as e:
        printer.error(f"❌ Error al obtener el feedback del vector store: {e}")
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
