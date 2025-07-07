import traceback

from server.celery_app import celery
from server.utils.printer import Printer
from server.utils.processor import (
    generate_sentence_brief,
    update_sentence_brief,
    sequencial_extraction,
    generate_feedback_from_messages,
)
from server.utils.csv_logger import CSVLogger

# from server.ai.ai_interface import tokenize_prompt


printer = Printer(name="tasks")
csv_logger = CSVLogger("tasks_log.csv")


def cut_user_message(previous_messages: list[dict], n_characters_to_cut: int):
    for message in previous_messages:
        if message["role"] == "user":
            message["content"] = message["content"][:-n_characters_to_cut]
    return previous_messages


@celery.task(
    name="extractor",
    autoretry_for=(Exception,),
    retry_kwargs={"countdown": 10},
    retry_backoff=True,
    bind=True,
)
def extractor_task(self, source_hash: str):
    task_name = "extractor"
    try:
        printer.info(f"Empezando a extraer datos del documento, HASH: {source_hash}")
        sequencial_extraction(source_hash)
        printer.info(
            f"Extracción de datos completada, empezando a generar la interpretación de la sentencia ciudadana, HASH: {source_hash}"
        )
        csv_logger.log(
            endpoint=task_name,
            http_status=200,
            hash_=source_hash,
            message="Extracción de datos completada, empezando a generar la interpretación de la sentencia ciudadana",
            exit_status=0,
        )
        generate_brief_task.delay(source_hash)
        return "Extracción de datos completada, empezando a generar la interpretación de la sentencia ciudadana"
    except Exception as e:
        printer.error("Error extrayendo el texto de origen:", e)
        tb = traceback.format_exc()
        printer.error(tb)
        csv_logger.log(
            endpoint=task_name,
            http_status=500,
            hash_=source_hash,
            message=str(tb),
            exit_status=1,
        )
        raise


@celery.task(
    name="generate_sentence_brief",
    autoretry_for=(Exception,),
    retry_kwargs={"countdown": 10},
    retry_backoff=True,
    bind=True,
    max_retries=5,
)
def generate_brief_task(
    self,
    source_hash: str,
) -> dict:
    task_name = "generate_sentence_brief"
    # N_CHARACTERS_TO_CUT = 5000
    # is_first_attempt = self.request.retries == 0
    task_traceback = ""
    try:
        generate_sentence_brief(source_hash)
        csv_logger.log(
            endpoint=task_name,
            http_status=200,
            hash_=source_hash,
            message="Resumen generado correctamente",
            exit_status=0,
        )
        return "Resumen generado correctamente"
    except Exception as e:
        printer.error("Error generando una sentencia ciudadana:", e)
        tb = traceback.format_exc()
        task_traceback += f"Error generando una sentencia ciudadana: {e}\n"
        task_traceback += f"Traceback: {tb}\n"
        printer.error(tb)
        csv_logger.log(
            endpoint=task_name,
            http_status=500,
            hash_=source_hash,
            message=str(task_traceback),
            exit_status=1,
        )
        raise


@celery.task(
    name="update_sentence_brief",
    autoretry_for=(Exception,),
    retry_kwargs={"countdown": 10},
    retry_backoff=True,
    bind=True,
    max_retries=5,
)
def update_brief_task(
    self, sources_hash: str, sentence: str, changes: str, prev_messages: str
) -> dict:
    task_name = "update_sentence_brief"
    try:
        printer.info(
            f"Actualizando la sentencia ciudadana, HASH: {sources_hash}, cambios: {changes}"
        )
        result = update_sentence_brief(sources_hash, sentence, changes, prev_messages)
        printer.debug("Resumen actualizado: ", result)
        csv_logger.log(
            endpoint=task_name,
            http_status=200,
            hash_=sources_hash,
            message="Resumen actualizado correctamente",
            exit_status=0,
        )
        return "Resumen actualizado correctamente"
    except Exception as e:
        tb = traceback.format_exc()
        printer.error("Error actualizando una sentencia ciudadana:", e)
        printer.error(tb)
        csv_logger.log(
            endpoint=task_name,
            http_status=500,
            hash_=sources_hash,
            message=str(tb),
            exit_status=1,
        )
        raise


@celery.task(
    name="generate_feedback",
    autoretry_for=(Exception,),
    retry_kwargs={"countdown": 10},
    retry_backoff=True,
    bind=True,
    max_retries=5,
)
def generate_feedback_task(self, sources_hash: str, messages: str):
    task_name = "generate_feedback"
    try:
        result = generate_feedback_from_messages(sources_hash, messages)
        printer.debug("Feedback generado: ", result)
        csv_logger.log(
            endpoint=task_name,
            http_status=200,
            hash_=sources_hash,
            message="Feedback generado correctamente",
            exit_status=0,
        )
        return "Feedback generado correctamente"
    except Exception as e:
        tb = traceback.format_exc()
        printer.error("Error generando feedback:", e)
        printer.error(tb)
