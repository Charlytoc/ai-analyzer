# utils/document_reader.py

from abc import ABC, abstractmethod
import os
import hashlib
import fitz  # PyMuPDF
import pytesseract
import pdfplumber
from PIL import Image
import io
from docx import Document

PAGE_CONNECTOR = "\n---PAGE---\n"


# =========================
# Estrategia base
# =========================

class DocumentStrategy(ABC):
    document_hash: str | None = None

    @abstractmethod
    def read(self, path: str) -> str:
        pass

    def hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def split_pages(self, text: str) -> list[str]:
        return text.split(PAGE_CONNECTOR)


# =========================
# Estrategias específicas
# =========================

class PyMuPDFWithOCRStrategy(DocumentStrategy):
    def read(self, path: str) -> str:
        pages = []
        with fitz.open(path, filetype="pdf") as pdf:
            for page in pdf:
                text = page.get_text()
                if not text.strip():
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    text = pytesseract.image_to_string(img)
                pages.append(text)
        return PAGE_CONNECTOR.join(pages)


class DocxStrategy(DocumentStrategy):
    def read(self, path: str) -> str:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)


class MarkdownStrategy(DocumentStrategy):
    def read(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


# =========================
# Lector de documentos
# =========================

class DocumentReader:
    text: str | None = None

    def __init__(self):
        self.strategy: DocumentStrategy | None = None

    def _get_strategy(self, path: str) -> DocumentStrategy:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            return PyMuPDFWithOCRStrategy()
        elif ext == ".docx":
            return DocxStrategy()
        elif ext == ".md":
            return MarkdownStrategy()
        else:
            raise ValueError(f"Tipo de archivo '{ext}' no soportado")

    def read(self, path: str) -> str:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        self.strategy = self._get_strategy(path)
        self.text = self.strategy.read(path)

        return self.text

    def split_pages(self, text: str) -> list[str]:
        if self.strategy is None:
            raise ValueError("No se ha leído ningún documento todavía")
        return self.strategy.split_pages(text)

    def get_hash(self) -> str:
        if self.text is None:
            raise ValueError("No se ha leído ningún documento todavía")
        return self.strategy.hash_text(self.text)
