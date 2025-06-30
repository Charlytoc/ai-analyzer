import re


def remove_h2_h3_h4_questions(text: str) -> str:
    # Elimina líneas que sean solo un encabezado (##, ###, ####) seguido de una pregunta, opcionalmente en negrita
    pattern = r"^(#{2,4})\s*(\*\*|__)?\s*¿[^?]+\?\s*(\*\*|__)?\s*$"
    return re.sub(pattern, "", text, flags=re.MULTILINE).strip()


texto = """
# Algun titulo

Intro

### ¿pregunta indeseada?

## **¿Otra pregunta?**

#### __¿Una más?__

Texto que debe quedar.
"""

print(remove_h2_h3_h4_questions(texto))
