import traceback

from server.celery_app import celery
from server.utils.printer import Printer
from server.utils.processor import (
    generate_sentence_brief,
    update_sentence_brief,
    format_response,
)
from server.utils.csv_logger import CSVLogger
from datetime import datetime

printer = Printer(name="tasks")
csv_logger = CSVLogger("tasks_log.csv")


@celery.task(
    name="generate_sentence_brief",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 10},
    retry_backoff=True,
)
def generate_brief_task(
    messages: list, messages_hash: str, n_documents: int, n_images: int
) -> dict:
    task_name = "generate_sentence_brief"
    timestamp = datetime.utcnow().isoformat()
    try:
        printer.info("Procesando mensajes para generar una sentencia ciudadana")
        sentence_brief = generate_sentence_brief(messages, messages_hash)
        resumen = format_response(
            sentence_brief, False, messages_hash, n_documents, n_images
        )
        printer.debug("Resumen generado: ", resumen)
        csv_logger.log(
            endpoint=task_name,
            http_status=200,
            hash_=messages_hash,
            message="Resumen generado correctamente",
            exit_status=0,
        )
        return "Resumen generado correctamente"
    except Exception as e:
        printer.error("Error en tarea:", e)
        tb = traceback.format_exc()
        printer.error(tb)
        csv_logger.log(
            endpoint=task_name,
            http_status=500,
            hash_=messages_hash,
            message=str(tb),
            exit_status=1,
        )
        raise


@celery.task(
    name="update_sentence_brief",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 10},
    retry_backoff=True,
)
def update_brief_task(messages_hash: str, changes: str) -> dict:
    task_name = "update_sentence_brief"
    try:
        result = update_sentence_brief(messages_hash, changes)
        printer.debug("Resumen actualizado: ", result)
        csv_logger.log(
            endpoint=task_name,
            http_status=200,
            hash_=messages_hash,
            message="Resumen actualizado correctamente",
            exit_status=0,
        )
        return "Resumen actualizado correctamente"
    except Exception as e:
        tb = traceback.format_exc()
        printer.error("Error en tarea:", e)
        printer.error(tb)
        csv_logger.log(
            endpoint=task_name,
            http_status=500,
            hash_=messages_hash,
            message=str(tb),
            exit_status=1,
        )
        raise
