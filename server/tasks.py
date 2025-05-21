from server.celery_app import celery
from server.utils.printer import Printer
from server.utils.processor import (
    generate_sentence_brief,
    update_sentence_brief,
    format_response,
)

printer = Printer(name="tasks")


@celery.task(name="generate_sentence_brief")
def generate_brief_task(
    messages: list, messages_hash: str, n_documents: int, n_images: int
) -> dict:
    printer.info("Procesando mensajes para generar una sentencia ciudadana")

    sentence_brief = generate_sentence_brief(messages, messages_hash)

    resumen = format_response(
        sentence_brief, False, messages_hash, n_documents, n_images
    )

    printer.debug("Resumen generado: ", resumen)

    return "Resumen generado correctamente"


@celery.task(name="update_sentence_brief")
def update_brief_task(messages_hash: str, changes: str) -> dict:
    result = update_sentence_brief(messages_hash, changes)
    printer.debug("Resumen actualizado: ", result)
    return "Resumen actualizado correctamente"
