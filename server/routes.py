import json
import traceback
from fastapi import APIRouter, UploadFile, File, Form
from starlette.concurrency import run_in_threadpool

import shutil
import os

from pydantic import BaseModel

# import numpy as np
from typing import List
from fastapi import HTTPException

from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.utils.processor import (
    generate_sentence_brief,
    update_sentence_brief,
    upsert_feedback_in_vector_store,
)
from server.utils.ai_interface import get_warning_text
from server.ai.vector_store import chroma_client


UPLOADS_PATH = "uploads"
os.makedirs(f"{UPLOADS_PATH}/images", exist_ok=True)
os.makedirs(f"{UPLOADS_PATH}/documents", exist_ok=True)

router = APIRouter(prefix="/api")
printer = Printer("ROUTES")
redis_cache = RedisCache()


@router.get("/sentencia/{hash}")
async def get_sentence_brief_route(hash: str):

    sentencia = redis_cache.get(f"sentence_brief:{hash}")
    if not sentencia:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "ERROR",
                "message": "No se encontró la sentencia ciudadana.",
            },
        )

    return {
        "status": "SUCCESS",
        "message": "Sentencia ciudadana generada con éxito.",
        "sentence": sentencia,
    }


class SentenceUpdatePayload(BaseModel):
    sentence: str


@router.put("/sentencia/{hash}")
async def update_sentence_brief_route(hash: str, payload: SentenceUpdatePayload):
    try:
        redis_cache.set(f"sentence_brief:{hash}", payload.sentence)
        return {
            "status": "SUCCESS",
            "message": "Sentencia ciudadana actualizada con éxito.",
            "sentence": payload.sentence,
        }
    except Exception as e:
        printer.error(f"❌ Error al actualizar la sentencia ciudadana: {e}")
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
        sentence = await run_in_threadpool(update_sentence_brief, hash, payload.changes)

        upsert_feedback_in_vector_store(hash, payload.changes)
        return {
            "status": "SUCCESS",
            "message": "Cambios realizados con éxito.",
            "changes": payload.changes,
            "sentence": sentence,
        }
    except Exception as e:
        printer.error(f"❌ Error al solicitar cambios de una sentencia: {e}")
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
        printer.yellow("🔄 Generando sentencia ciudadana...")

        print("🚨 Validación de archivos", chroma_client)
        # 🚨 Validación de archivos
        if not images and not documents:
            printer.error(
                "Debes enviar al menos un documento o una imagen para generar la sentencia ciudadana."
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

        printer.yellow("🔄 Generando sentencia ciudadana en otro hilo...")
        resumen, cache_used, hash_messages = await run_in_threadpool(
            generate_sentence_brief, document_paths, images_paths, extra_info
        )

        if cache_used:
            printer.magenta("🔄 Sentencia ciudadana generada con caché")
            printer.green(resumen)
        else:
            printer.magenta("✅ Sentencia ciudadana generada sin caché")
            printer.green(resumen)

        return {
            "status": "SUCCESS",
            "message": "Sentencia ciudadana generada con éxito.",
            "brief": resumen,
            "n_documents": len(document_paths),
            "n_images": len(images_paths),
            "cache_used": cache_used,
            "hash": hash_messages,
            "warning": get_warning_text(),
        }
    except Exception as e:
        tb = traceback.format_exc()
        printer.error(f"❌ Error al generar la sentencia ciudadana: {e}")
        printer.error(tb)
        raise HTTPException(
            status_code=500,
            detail={"status": "ERROR", "message": str(e)},
        )
