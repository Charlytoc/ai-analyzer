import csv
import os
import threading
from datetime import datetime

class InteractionLogger:
    def __init__(self, base_name="interactions"):
        self.base_name = base_name
        self.log_dir = "logs"
        self.lock = threading.Lock()
        self.fields = [
            "username",
            "hash",
            "message",
            "timestamp",
        ]
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_file_path(self):
        today_str = datetime.utcnow().strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"{self.base_name}_{today_str}.csv")

    def log(self, username, hash_, message):
        row = {
            "username": username,
            "hash": hash_,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        file_path = self._get_file_path()
        with self.lock:
            file_exists = os.path.exists(file_path)
            with open(file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
