import os
import hashlib
import json

from server.utils.pdf_reader import DocumentReader
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.utils.ai_interface import AIInterface, get_physical_context
from server.utils.image_reader import ImageReader
from server.ai.vector_store import chroma_client

EXPIRATION_TIME = 60 * 60 * 24 * 30  # 30 days

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", None)
LIMIT_CHARACTERS_FOR_FILE = 50000

printer = Printer("ROUTES")
redis_cache = RedisCache()


def hasher(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_faq_questions():
    return [
        "¿Cuál es el número de expediente?",
        "¿Qué tribunal emitió la sentencia?",
        "¿Quiénes son las partes involucradas en el juicio?",
        "¿Qué tipo de juicio o procedimiento se trata?",
        "¿Qué hechos se narran como antecedentes?",
        "¿Qué pruebas se ofrecieron en el juicio?",
        "¿Qué actos procesales relevantes se mencionan?",
        "¿Qué cuestiones jurídicas se analizan en los considerandos?",
        "¿Qué leyes o principios jurídicos se citaron?",
        "¿Cómo se valoraron las pruebas presentadas?",
        "¿Qué argumentos de las partes fueron aceptados o rechazados?",
        "¿Cuál fue la decisión final del juez?",
        "¿Qué órdenes específicas se dieron en los puntos resolutivos?",
        "¿Está incluida la firma y sello del juez y secretario?",
        "¿Se usó Firma Electrónica Judicial (FEJEM)?",
        "¿Qué preguntas clave del caso se resolvieron?",
        "¿Cómo se explicó la norma legal aplicada?",
        "¿Qué consecuencias o pasos a seguir se derivan de la sentencia?",
        "¿El lenguaje utilizado es claro y comprensible para ciudadanos?",
        "¿Se explican los tecnicismos jurídicos usados?",
    ]


def flatten_list(nested_list):
    """Aplana una lista de listas en una sola lista de elementos."""
    if not nested_list:
        return []
    return [item for sublist in nested_list for item in sublist]


def get_faq_results(doc_hash: str):
    results_str = ""

    for question in get_faq_questions():
        retrieval = chroma_client.get_results(
            collection_name=f"doc_{doc_hash}",
            query_texts=[question],
            n_results=3,
        )
        documents = flatten_list(retrieval.get("documents", []))
        results_str += f" {question.upper()}{' '.join(documents)}"

    return results_str


def translate_to_spanish(text: str):
    system_prompt = """
    Your task is to translate a given text to spanish, preserve the original meaning and structure of the text. Return only the translated text, without any other text or explanation.
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
    system_from_client = extra.get("system_prompt", None)
    ai_interface = AIInterface(
        provider=os.getenv("PROVIDER", "ollama"), api_key=os.getenv("OLLAMA_API_KEY")
    )

    system_prompt = (
        system_from_client
        or SYSTEM_PROMPT
        or """
    ### Role
    You are an incredible legal expert in Mexican law. You are will receive all the possible text from a set of documents and images, you need to analyze them, they are related to the same case. Extract the relevant information and explain the result of the sentence in a clear and concise way,  any person should understand it. Keep in mind that your main task is to write a "Sentencia Ciudadana". You will receive a set of context documents with an example of a "Sentencia Ciudadana" and some guidelines for you to follow.

    ### Context
    These files are part of the context given to execute your task, use them to comprehend your task and follow the writing guidelines explained in the files:
    ---
    {{context}}
    ---

    ### Instructions
    - Write a "Sentencia Ciudadana" using the information provided in the user messages.
    - The information provided in the user messages is the result of a retrieval of the most relevant information from the files using a vector database, don't miss any important detail.
    - Your response must be ONLY in Spanish.
    - The final response should include all the relevant information from the files, don't miss any important detail.
    """
    )
    if physical_context:
        system_prompt = system_prompt.replace("{{context}}", physical_context)

    messages = [{"role": "system", "content": system_prompt}]
    document_reader = DocumentReader()

    for document_path in document_paths:
        document_text = document_reader.read(document_path)
        document_hash = hasher(document_text)
        chroma_client.get_or_create_collection(f"doc_{document_hash}")
        chunks = chroma_client.chunkify(
            document_text, chunk_size=1000, chunk_overlap=200
        )
        chroma_client.bulk_upsert_chunks(
            collection_name=f"doc_{document_hash}",
            chunks=chunks,
        )
        faq_results = get_faq_results(document_hash)
        messages.append(
            {
                "role": "user",
                "content": f"# FAQ RESULTS FOR DOCUMENT, use this information to write the Sentencia Ciudadana: {document_path}\n\n{faq_results}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"Initial content of the document, often it contains a lot of useful information to extract: {document_text[:15000]}",
            }
        )

    for image_path in images_paths:
        image_reader = ImageReader()
        image_text = image_reader.read(image_path)

        messages.append(
            {
                "role": "user",
                "content": f"<Image source={image_path}>\n\n{image_text}\n\n</Image>",
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
            printer.green(f"Cache hit for hash: {messages_hash}")
            return cached_response

    printer.red(f"Cache miss for hash: {messages_hash}")
    response = ai_interface.chat(messages=messages, model=os.getenv("MODEL", "gemma3"))
    response = translate_to_spanish(response)

    redis_cache.set(messages_hash, response, ex=EXPIRATION_TIME)
    printer.green(f"Cache saved for hash: {messages_hash}")

    return response
