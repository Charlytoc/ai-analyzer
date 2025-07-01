import traceback
import json

from server.celery_app import celery
from server.utils.printer import Printer
from server.utils.processor import (
    generate_sentence_brief,
    update_sentence_brief,
    format_response,
)
from server.utils.csv_logger import CSVLogger
from server.ai.ai_interface import tokenize_prompt


printer = Printer(name="tasks")
csv_logger = CSVLogger("tasks_log.csv")


def cut_user_message(previous_messages: list[dict], n_characters_to_cut: int):
    for message in previous_messages:
        if message["role"] == "user":
            message["content"] = message["content"][:-n_characters_to_cut]
    return previous_messages


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
    messages: list,
    messages_hash: str,
    n_documents: int,
    n_images: int,
) -> dict:
    task_name = "generate_sentence_brief"
    N_CHARACTERS_TO_CUT = 5000
    is_first_attempt = self.request.retries == 0
    task_traceback = ""
    try:
        printer.info(
            f"Procesando mensajes para generar una sentencia ciudadana, intento #{self.request.retries} de {self.max_retries}"
        )
        task_traceback += f"Procesando mensajes para generar una sentencia ciudadana, intento #{self.request.retries} de {self.max_retries}\n"
        if not is_first_attempt:
            printer.info(
                f"Cortando mensajes para reintentar la generación de una sentencia ciudadana con menos caracteres, intento #{self.request.retries} de {self.max_retries}"
            )
            task_traceback += f"Cortando mensajes para reintentar la generación de una sentencia ciudadana con menos caracteres, intento #{self.request.retries} de {self.max_retries}\n"

            task_traceback += (
                f"Antes de cortar: {len(json.dumps(messages))} caracteres\n"
            )
            messages = cut_user_message(
                messages, N_CHARACTERS_TO_CUT * (self.request.retries)
            )
            # print(len(json.dumps(messages)), "characters after cut")
            task_traceback += (
                f"Después de cortar: {len(json.dumps(messages))} caracteres\n"
            )

        prompt = messages[-1]["content"] + " " + messages[-2]["content"]

        count, difference, is_difference_more_than_4000 = tokenize_prompt(prompt)
        if not is_difference_more_than_4000:
            printer.error(
                f"El prompt {len(prompt)} caracteres es demasiado largo y resulta en un prompt de {count} tokens, la diferencia es de {difference} tokens. Cortando el prompt."
            )
            task_traceback += f"El prompt {len(prompt)} caracteres es demasiado largo y resulta en un prompt de {count} tokens, la diferencia es de {difference} tokens. Cortando el prompt.\n"
            messages = cut_user_message(
                messages, N_CHARACTERS_TO_CUT * (self.request.retries + 1)
            )

        sentence_brief = generate_sentence_brief(messages, messages_hash)
        resumen = format_response(
            sentence_brief, False, messages_hash, n_documents, n_images
        )
        printer.yellow("Resumen generado: ", resumen)
        csv_logger.log(
            endpoint=task_name,
            http_status=200,
            hash_=messages_hash,
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
            hash_=messages_hash,
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
def update_brief_task(self, messages_hash: str, sentence: str, changes: str) -> dict:
    task_name = "update_sentence_brief"
    try:
        result = update_sentence_brief(messages_hash, sentence, changes)
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
        printer.error("Error actualizando una sentencia ciudadana:", e)
        printer.error(tb)
        csv_logger.log(
            endpoint=task_name,
            http_status=500,
            hash_=messages_hash,
            message=str(tb),
            exit_status=1,
        )
        raise
