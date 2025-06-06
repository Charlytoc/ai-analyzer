import json

import traceback
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

import shutil
import os

from pydantic import BaseModel

from typing import List
from fastapi import HTTPException
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.utils.processor import (
    format_messages,
    upsert_feedback_in_vector_store,
    hasher,
    EXPIRATION_TIME,
    format_response,
    generate_sentence_brief,
)
from server.ai.ai_interface import get_warning_text
from server.utils.csv_logger import CSVLogger
from server.tasks import generate_brief_task, update_brief_task

csv_logger = CSVLogger()

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
    return JSONResponse(
        content={
            "status": "SUCCESS",
            "message": "Sentencia ciudadana generada con éxito.",
            "brief": sentencia,
            "hash": hash,
            "warning": get_warning_text(),
        },
        status_code=200,
    )


class SentenceUpdatePayload(BaseModel):
    sentence: str


@router.put("/sentencia/{hash}")
async def update_sentence_brief_route(hash: str, payload: SentenceUpdatePayload):
    try:
        redis_cache.set(f"sentence_brief:{hash}", payload.sentence)

        csv_logger.log(
            "PUT /sentencia/{hash}",
            200,
            hash,
            "Sentencia ciudadana actualizada con éxito.",
            exit_status=0,
        )
        return {
            "status": "SUCCESS",
            "message": "Sentencia ciudadana actualizada con éxito.",
            "sentence": payload.sentence,
        }
    except Exception as e:
        printer.error(f"❌ Error al actualizar la sentencia ciudadana: {e}")
        csv_logger.log(
            "PUT /sentencia/{hash}",
            500,
            hash,
            f"Error al actualizar la sentencia ciudadana: {e}",
            exit_status=1,
        )
        raise HTTPException(
            status_code=500,
            detail={"status": "ERROR", "message": str(e)},
        )


class SentenceRequestChangesPayload(BaseModel):
    changes: str


@router.post("/sentencia/{hash}/request-changes")
async def request_changes_route(hash: str, payload: SentenceRequestChangesPayload):

    try:
        sentence = redis_cache.get(f"sentence_brief:{hash}")
        if not sentence:
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "ERROR",
                    "message": "No se encontró la sentencia ciudadana.",
                },
            )

        printer.yellow("🔄 Actualizando sentencia ciudadana en otro hilo...")

        update_brief_task.delay(hash, payload.changes)

        upsert_feedback_in_vector_store(hash, payload.changes)

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


@router.post("/generate-sentence-brief")
async def generate_sentence_brief_route(
    images: List[UploadFile] = File([]),
    documents: List[UploadFile] = File([]),
    extra_data: str = Form(None),
):
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

        # Procesar el JSON si viene
        extra_info = {}
        if extra_data:
            try:
                extra_info = json.loads(extra_data)
                printer.blue(extra_info, "Información adicional recibida")
            except json.JSONDecodeError:
                printer.error("❌ Error al decodificar el JSON enviado en extra_data")

        use_cache = extra_info.get("use_cache", True)
        process_async = extra_info.get("async", True)
        messages = format_messages(document_paths, images_paths)

        messages_json = json.dumps(messages, sort_keys=True, indent=4)
        messages_hash = hasher(messages_json)

        redis_cache.set(
            f"messages_input:{messages_hash}", messages_json, ex=EXPIRATION_TIME
        )

        if use_cache:
            cached_response = redis_cache.get(f"sentence_brief:{messages_hash}")
            if cached_response:
                printer.green(f"👀 Sentencia ciudadana cacheada: {messages_hash}")

                response = format_response(
                    cached_response,
                    True,
                    messages_hash,
                    len(document_paths),
                    len(images_paths),
                )
                csv_logger.log(
                    "POST /generate-sentence-brief",
                    200,
                    messages_hash,
                    "Sentencia ciudadana obtenida de caché.",
                    exit_status=0,
                )
                return response

        printer.yellow(
            f"🔍 No se encontró la sentencia ciudadana en cache: {messages_hash}"
        )

        for image_path in images_paths:
            os.remove(image_path)
        for document_path in document_paths:
            os.remove(document_path)

        if process_async:
            printer.yellow(
                f"🔄 Enviando tarea de generación de sentencia ciudadana a cola de tareas, HASH: {messages_hash}"
            )
            generate_brief_task.delay(
                messages,
                messages_hash,
                len(document_paths),
                len(images_paths),
            )
        else:
            result = generate_sentence_brief(
                messages,
                messages_hash,
            )

            redis_cache.set(
                f"sentence_brief:{messages_hash}", result, ex=EXPIRATION_TIME
            )

            csv_logger.log(
                "POST /generate-sentence-brief",
                200,
                messages_hash,
                "Sentencia ciudadana generada con éxito de forma síncrona.",
                exit_status=0,
            )
            return JSONResponse(
                content=format_response(
                    result,
                    False,
                    messages_hash,
                    len(document_paths),
                    len(images_paths),
                ),
                status_code=200,
            )
        printer.green(
            f"Sentencia ciudadana en proceso de generación en segundo plano, HASH: {messages_hash}"
        )

        csv_logger.log(
            "POST /generate-sentence-brief",
            201,
            messages_hash,
            "Sentencia ciudadana en cola...",
            exit_status=0,
        )

        return JSONResponse(
            content={
                "status": "QUEUED",
                "message": "Sentencia ciudadana en cola...",
                "hash": messages_hash,
            },
            status_code=201,
        )
    except Exception as e:
        tb = traceback.format_exc()
        printer.error(f"❌ Error al generar la sentencia ciudadana: {e}")
        printer.error(tb)
        raise HTTPException(
            status_code=500,
            detail={"status": "ERROR", "message": str(e)},
        )
