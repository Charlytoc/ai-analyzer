import csv
import os
import threading
from datetime import datetime


class CSVLogger:
    def __init__(self, file_path="requests_log.csv"):
        self.file_path = file_path
        self.lock = threading.Lock()
        self.fields = [
            "timestamp",
            "endpoint",
            "http_status",
            "hash",
            "message",
            "exit_status",
        ]
        # Escribir cabecera solo si el archivo no existe
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log(self, endpoint, http_status, hash_, message, exit_status: int = 0):
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "http_status": http_status,
            "hash": hash_,
            "message": message,
            "exit_status": exit_status,
        }
        with self.lock:
            with open(self.file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writerow(row)
