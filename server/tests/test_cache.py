# test_cache.py
import pytest
from functools import lru_cache
from server.utils import ai_interface
from server.utils.ai_interface import get_faq_questions, get_system_prompt

@pytest.fixture(autouse=True)
def clear_caches():
    # Limpia el cache antes de cada test
    get_system_prompt.cache_clear()
    get_faq_questions.cache_clear()


def test_get_system_prompt():
    prompt = get_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_get_faq_questions():
    questions = get_faq_questions()
    assert isinstance(questions, list)
    assert all(isinstance(q, str) for q in questions)
    assert len(questions) > 0


def test_get_system_prompt_cache(monkeypatch):
    call_counter = {"count": 0}

    def mock_prompt():
        call_counter["count"] += 1
        return "mocked prompt"

    monkeypatch.setattr(ai_interface, "get_system_prompt", lru_cache()(mock_prompt))

    ai_interface.get_system_prompt()
    ai_interface.get_system_prompt()
    ai_interface.get_system_prompt()

    assert call_counter["count"] == 1


def test_get_faq_questions_cache(monkeypatch):
    call_counter = {"count": 0}

    def mock_faq():
        call_counter["count"] += 1
        return ["Q1", "Q2"]

    monkeypatch.setattr(ai_interface, "get_faq_questions", lru_cache()(mock_faq))

    ai_interface.get_faq_questions()
    ai_interface.get_faq_questions()
    ai_interface.get_faq_questions()

    assert call_counter["count"] == 1
