import sys
from unittest.mock import MagicMock

# Mockea celery y ollama
sys.modules["celery"] = MagicMock()
sys.modules["ollama"] = MagicMock()
sys.modules["openai"] = MagicMock()
