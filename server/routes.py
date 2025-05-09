import json
from fastapi import APIRouter, UploadFile, File, Form
import shutil
import os

# import numpy as np
from typing import List
from fastapi import HTTPException

from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache
from server.utils.processor import generate_sentence_brief
from server.ai.vector_store import chroma_client

# from server.utils.ai_interface import AIInterface

UPLOADS_PATH = "uploads"
os.makedirs(f"{UPLOADS_PATH}/images", exist_ok=True)
os.makedirs(f"{UPLOADS_PATH}/documents", exist_ok=True)

router = APIRouter(prefix="/api")
printer = Printer("ROUTES")
redis_cache = RedisCache()


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.linalg.norm(a - b))


# @router.post("/compare-embeddings")
# async def compare_embeddings_route(body: dict = Body(...)):
#     target_text = body.get("target_text")
#     compares = body.get("compares", [])
#     func = body.get("function", "cosine").lower()

#     if not target_text or not compares or func not in {"cosine", "euclidean"}:
#         raise HTTPException(400, "Payload inválido.")

#     ai = AIInterface(provider="ollama", api_key="")
#     try:
#         # 1) Embed y aplanar a 1D
#         t_emb = np.array(ai.embed(target_text)["embeddings"]).flatten()
#         c_embs = [np.array(ai.embed(text)["embeddings"]).flatten() for text in compares]
#     except Exception as e:
#         raise HTTPException(500, f"Error al embed: {e}")

#     # 2) Elige función
#     comparator = cosine_similarity if func == "cosine" else euclidean_distance
#     metric = "cosine similarity" if func == "cosine" else "euclidean distance"

#     # 3) Calcula scores
#     results = []
#     for text, emb in zip(compares, c_embs):
#         score = comparator(t_emb, emb)
#         results.append({"text": text, "score": score})

#     return {"status": "OK", "metric": metric, "results": results}


# @router.post("/embed-text")
# async def embed_text_route(payload: dict = Body(...)):
#     text = payload.get("text")
#     ai_interface = AIInterface(provider="ollama", api_key="")
#     try:
#         result = ai_interface.embed(text)
#         vector = result["embeddings"]
#         # Save the vector result as vector.json
#         with open("vector.json", "w") as f:
#             json.dump(vector, f)
#         return {"status": "OK", "embedding": vector}
#     except Exception as e:
#         printer.red(f"❌ Error al generar el embedding: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail={"status": "ERROR", "message": str(e)},
#         )


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

        resumen, cache_used = generate_sentence_brief(
            document_paths, images_paths, extra_info
        )
        printer.green(f"✅ Sentencia ciudadana generada con éxito:\n{resumen}")

        return {
            "status": "SUCCESS",
            "message": "Sentencia ciudadana generada con éxito.",
            "brief": resumen,
            "n_documents": len(document_paths),
            "n_images": len(images_paths),
            "cache_used": cache_used,
        }
    except Exception as e:
        printer.red(f"❌ Error al generar la sentencia ciudadana: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "ERROR", "message": str(e)},
        )
