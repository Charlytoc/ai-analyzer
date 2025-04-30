import os
import hashlib
from abc import ABC, abstractmethod
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# =========================
# Configuración flexible
# =========================

load_dotenv()


tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    print("🔍 Usando tesseract_cmd:", tesseract_cmd)

    # Si es Windows, aseguramos que termina en tesseract.exe
    if os.name == "nt":
        if os.path.isdir(tesseract_cmd):
            tesseract_cmd = os.path.join(tesseract_cmd, "tesseract.exe")
        
        if not os.path.isfile(tesseract_cmd):
            raise FileNotFoundError(
                f"El ejecutable de tesseract no se encontró en: {tesseract_cmd}"
            )

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# =========================
# Estrategia base
# =========================


class ImageStrategy(ABC):
    @abstractmethod
    def read(self, path: str) -> str:
        pass

    def hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =========================
# Estrategias específicas
# =========================


class OCRImageStrategy(ImageStrategy):
    def read(self, path: str) -> str:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip()


# =========================
# Lector de imágenes
# =========================


class ImageReader:
    text: str | None = None

    def __init__(self):
        self.strategy: ImageStrategy = OCRImageStrategy()

    def read(self, path: str) -> str:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        self.text = self.strategy.read(path)
        return self.text

    def get_hash(self) -> str:
        if self.text is None:
            raise ValueError("No se ha leído ninguna imagen todavía")
        return self.strategy.hash_text(self.text)
