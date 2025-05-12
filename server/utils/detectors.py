from langdetect import detect, DetectorFactory, LangDetectException
from server.utils.printer import Printer

# Fijar la semilla para resultados consistentes
DetectorFactory.seed = 0


def is_spanish(texto):
    """
    Retorna True si el texto está en español, False en caso contrario.
    Si ocurre un error, se asume que el texto no está en español.
    """
    try:
        idioma = detect(texto)
        print(idioma, "idioma detectado")
        return idioma == "es"
    except LangDetectException:
        printer = Printer(name="LANG_DETECTOR")
        printer.red(
            "🔍 Error al detectar el idioma del texto, asumiendo que no está en español."
        )
        return False
