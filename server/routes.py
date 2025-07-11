import json

import traceback
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse

import shutil
import os

from pydantic import BaseModel

from typing import List
from fastapi import HTTPException
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.utils.processor import (
    read_sources,
    upsert_feedback_in_redis,
    hasher,
    EXPIRATION_TIME,
    validate_attachments,
)
from server.ai.ai_interface import get_warning_text
from server.utils.csv_logger import CSVLogger
from server.utils.interaction_logger import InteractionLogger
from server.tasks import update_brief_task, extractor_task, generate_feedback_task


csv_logger = CSVLogger()
interaction_logger = InteractionLogger()

DEFAULT_CACHE_BEHAVIOR = os.environ.get("DEFAULT_CACHE_BEHAVIOR", "false")
if DEFAULT_CACHE_BEHAVIOR.lower().strip() == "true":
    DEFAULT_CACHE_BEHAVIOR = True
else:
    DEFAULT_CACHE_BEHAVIOR = False


UPLOADS_PATH = "uploads"
os.makedirs(f"{UPLOADS_PATH}/images", exist_ok=True)
os.makedirs(f"{UPLOADS_PATH}/documents", exist_ok=True)

router = APIRouter(prefix="/api")
printer = Printer("ROUTES")
redis_cache = RedisCache()


@router.get("/sentencia/{hash}")
async def get_sentence_brief_route(hash: str):

    printer.yellow(f"🔄 Buscando sentencia ciudadana en cache: {hash}")
    sentencia = redis_cache.get(f"sentence_brief:{hash}")

    if not sentencia:
        csv_logger.log(
            "GET /sentencia/{hash}",
            404,
            hash,
            "No se encontró la sentencia ciudadana.",
            exit_status=1,
        )
        raise HTTPException(
            status_code=404,
            detail={
                "status": "ERROR",
                "message": "No se encontró la sentencia ciudadana.",
            },
        )

    csv_logger.log(
        "GET /sentencia/{hash}",
        200,
        hash,
        "Sentencia ciudadana encontrada en caché.",
        exit_status=0,
    )

    redis_cache.delete(f"sentence_brief:{hash}")
    sentece_json = json.loads(sentencia)

    return JSONResponse(
        content={
            "status": "SUCCESS",
            "message": sentece_json.get(
                "message", "Sentencia ciudadana encontrada en caché."
            ),
            "workflow": sentece_json.get("workflow", "update"),
            "brief": sentece_json.get("sentence", ""),
            "hash": hash,
            "warning": get_warning_text(),
            "storage_status": "DELETED",
            "rejected": sentece_json.get("rejected", False),
        },
        status_code=200,
    )


class SentenceRequestChangesPayload(BaseModel):
    sentence: str
    changes: str
    prev_messages: str


@router.post("/sentencia/{hash}/request-changes")
async def request_changes_route(
    hash: str, payload: SentenceRequestChangesPayload, request: Request
):
    try:
        username = request.headers.get("username")
        if not username:
            raise HTTPException(
                status_code=400,
                detail={"status": "ERROR", "message": "Falta el header 'username'."},
            )
        sentence = payload.sentence
        if not sentence:
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "ERROR",
                    "message": "No se encontró la sentencia ciudadana.",
                },
            )

        printer.yellow("🔄 Actualizando sentencia ciudadana en otro hilo...")

        update_brief_task.delay(hash, sentence, payload.changes, payload.prev_messages)

        interaction_logger.log(
            username=username,
            hash_=hash,
            message=payload.changes,
        )

        csv_logger.log(
            "POST /sentencia/{hash}/request-changes",
            201,
            hash,
            "Solicitud de cambios enviada con éxito.",
            exit_status=0,
        )
        return JSONResponse(
            content={
                "status": "QUEUED",
                "message": "Los cambios se aplicarán en unos minutos.",
                "changes": payload.changes,
                "hash": hash,
            },
            status_code=201,
        )
    except Exception as e:
        printer.error(f"❌ Error al solicitar cambios de una sentencia: {e}")
        csv_logger.log(
            "POST /sentencia/{hash}/request-changes",
            500,
            hash,
            f"Error al solicitar cambios de una sentencia: {e}",
            exit_status=1,
        )
        raise HTTPException(
            status_code=500,
            detail={"status": "ERROR", "message": str(e)},
        )


class FeedbackRequest(BaseModel):
    hash: str
    feedback: str


@router.post("/feedback")
async def feedback_route(payload: FeedbackRequest):
    try:
        upsert_feedback_in_redis(payload.feedback)
        return JSONResponse(
            content={
                "status": "SUCCESS",
                "message": "Feedback procesado con éxito.",
                "hash": payload.hash,
            },
            status_code=200,
        )
    except Exception as e:
        printer.error(f"❌ Error al procesar el feedback: {e}")
        raise HTTPException(
            status_code=500, detail={"status": "ERROR", "message": str(e)}
        )


@router.post("/generate-sentence-brief")
async def generate_sentence_brief_route(
    images: List[UploadFile] = File([]),
    documents: List[UploadFile] = File([]),
):
    tb = ""
    try:
        if not images and not documents:
            printer.error(
                "Debes enviar al menos un documento o una imagen para generar la sentencia ciudadana."
            )
            csv_logger.log(
                "POST /generate-sentence-brief",
                400,
                None,
                "Debes enviar al menos un documento o una imagen para generar la sentencia ciudadana.",
                exit_status=1,
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "ERROR",
                    "message": "Debes enviar al menos un documento o una imagen para generar la sentencia ciudadana.",
                },
            )

        images, documents = validate_attachments(images, documents)

        printer.yellow(f"🔍 Imágenes validas: {len(images)}")
        printer.yellow(f"🔍 Documentos validos: {len(documents)}")

        document_paths: list[str] = []
        images_paths: list[str] = []

        for image in images:
            image_path = f"{UPLOADS_PATH}/images/{image.filename}"
            images_paths.append(image_path)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

        for document in documents:
            document_path = f"{UPLOADS_PATH}/documents/{document.filename}"
            document_paths.append(document_path)
            with open(document_path, "wb") as buffer:
                shutil.copyfileobj(document.file, buffer)

        complete_text = read_sources(document_paths, images_paths)
        source_hash = hasher(complete_text)

        redis_cache.set(f"source_text:{source_hash}", complete_text, ex=EXPIRATION_TIME)

        for image_path in images_paths:
            try:
                os.remove(image_path)
            except Exception as e:
                tb += f"❌ Error al eliminar la imagen: {e}\n"
                printer.error(f"❌ Error al eliminar la imagen: {e}")
        for document_path in document_paths:
            try:
                os.remove(document_path)
            except Exception as e:
                tb += f"❌ Error al eliminar el documento: {e}\n"
                printer.error(f"❌ Error al eliminar el documento: {e}")

        printer.yellow(
            f"🔄 Enviando tarea de generación de sentencia ciudadana a cola de tareas, HASH: {source_hash}"
        )
        extractor_task.delay(source_hash)

        printer.green(
            f"Sentencia ciudadana en proceso de generación en segundo plano, HASH: {source_hash}"
        )
        message_to_log = (
            "Sentencia ciudadana en cola..." + "\nTRACEBACK: " + tb
            if tb
            else "Sentencia ciudadana en cola..."
        )

        csv_logger.log(
            "POST /generate-sentence-brief",
            201,
            source_hash,
            message_to_log,
            exit_status=0,
        )

        return JSONResponse(
            content={
                "status": "QUEUED",
                "message": "Sentencia ciudadana en cola...",
                "hash": source_hash,
                "text_from_all_documents": complete_text,
            },
            status_code=201,
        )
    except Exception as e:
        tb = traceback.format_exc()
        printer.error(f"❌ Error al generar la sentencia ciudadana: {e}")
        printer.error(tb)
        printer.error("🔍 Eliminando archivos temporales...")

        for image_path in images_paths:
            try:
                os.remove(image_path)
            except Exception as e:
                tb += f"❌ Error al eliminar la imagen: {e}\n"
                tb += "Borrar manualmente si es necesario\n"
                printer.error(f"❌ Error al eliminar la imagen: {e}")
        for document_path in document_paths:
            try:
                os.remove(document_path)
            except Exception as e:
                tb += f"❌ Error al eliminar el documento: {e}\n"
                tb += "Borrar manualmente si es necesario\n"
                printer.error(
                    f"❌ Error al eliminar el documento: {e} Borrar manualmente si es necesario"
                )

        raise HTTPException(
            status_code=500,
            detail={"status": "ERROR", "message": str(e)},
        )


class FeedbackGenerateRequest(BaseModel):
    hash: str
    messages: str


@router.post("/generate-feedback")
async def generate_feedback_route(payload: FeedbackGenerateRequest):
    try:
        printer.yellow(
            f"🔄 Enviando tarea de generación de feedback a cola de tareas, HASH: {payload.hash}"
        )
        generate_feedback_task.delay(payload.hash, payload.messages)
        return JSONResponse(
            content={
                "status": "QUEUED",
                "message": "Feedback en cola...",
                "hash": payload.hash,
            },
        )
    except Exception as e:
        printer.error(f"❌ Error al generar el feedback: {e}")
        raise HTTPException(
            status_code=500, detail={"status": "ERROR", "message": str(e)}
        )


@router.get("/feedback/{hash}")
async def get_feedback_route(hash: str):
    try:
        feedback = redis_cache.get(f"feedback:{hash}")
        if not feedback:
            raise HTTPException(
                status_code=404,
                detail={"status": "ERROR", "message": "No se encontró el feedback."},
            )
        redis_cache.delete(f"feedback:{hash}")
        return JSONResponse(
            content={
                "status": "SUCCESS",
                "message": "Feedback obtenido con éxito.",
                "feedback": feedback,
            },
            status_code=200,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        printer.error(f"❌ Error al obtener el feedback: {e}")
        raise HTTPException(
            status_code=500, detail={"status": "ERROR", "message": str(e)}
        )
