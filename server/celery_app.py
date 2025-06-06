# server/celery_app.py
from celery import Celery
import os
import platform


# Config Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6380")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"


# Inicializa Celery
celery = Celery(
    "sentencia_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Configuración general
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# Ajuste dinámico del pool según el sistema operativo
if platform.system() == "Windows":
    # Celery no soporta prefork en Windows, usar threads
    celery.conf.worker_pool = "threads"
    celery.conf.worker_concurrency = int(os.getenv("CELERY_CONCURRENCY", "4"))
else:
    # En sistemas tipo Unix, usar prefork por defecto
    celery.conf.worker_pool = os.getenv("CELERY_POOL", "prefork")
    celery.conf.worker_concurrency = int(os.getenv("CELERY_CONCURRENCY", "8"))
