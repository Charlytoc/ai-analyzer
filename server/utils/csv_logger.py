import csv
import os
import threading
from datetime import datetime

class CSVLogger:
    def __init__(self, base_name="requests_log"):
        self.base_name = base_name
        self.log_dir = "logs"
        self.lock = threading.Lock()
        self.fields = [
            "timestamp",
            "endpoint",
            "http_status",
            "hash",
            "message",
            "exit_status",
        ]
        # Crear el directorio de logs si no existe
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_file_path(self):
        today_str = datetime.utcnow().strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"{self.base_name}_{today_str}.csv")

    def log(self, endpoint, http_status, hash_, message, exit_status: int = 0):
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "http_status": http_status,
            "hash": hash_,
            "message": message,
            "exit_status": exit_status,
        }
        file_path = self._get_file_path()
        with self.lock:
            file_exists = os.path.exists(file_path)
            with open(file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
