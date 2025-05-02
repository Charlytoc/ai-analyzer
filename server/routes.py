import json
from fastapi import APIRouter, UploadFile, File, Form
import shutil
import os
from typing import List
from fastapi import HTTPException

from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.utils.processor import generate_sentence_brief
from server.ai.vector_store import chroma_client

UPLOADS_PATH = "uploads"
os.makedirs(f"{UPLOADS_PATH}/images", exist_ok=True)
os.makedirs(f"{UPLOADS_PATH}/documents", exist_ok=True)

router = APIRouter(prefix="/api")
printer = Printer("ROUTES")
redis_cache = RedisCache()


@router.post("/generate-sentence-brief")
async def generate_sentence_brief_route(
    images: List[UploadFile] = File([]),
    documents: List[UploadFile] = File([]),
    extra_data: str = Form(None),
):
    printer.yellow("🔄 Generando sentencia ciudadana...")

    print("🚨 Validación de archivos", chroma_client)
    # 🚨 Validación de archivos
    if not images and not documents:
        printer.red("❌ No se enviaron documentos ni imágenes.")
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
            printer.red("❌ Error al decodificar el JSON enviado en extra_data")

    resumen = generate_sentence_brief(document_paths, images_paths, extra_info)
    printer.green(f"✅ Sentencia ciudadana generada con éxito:\n{resumen}")

    return {
        "status": "SUCCESS",
        "message": "Sentencia ciudadana generada con éxito.",
        "brief": resumen,
    }
