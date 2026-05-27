"""Tests for config.py (constants module — smoke check that values exist and are sane)."""

import config


def test_age_range():
    assert config.AGE_MIN == 7
    assert config.AGE_MAX == 14


def test_word_limits():
    assert config.STORY_WORD_MIN < config.STORY_WORD_MAX


def test_llm_defaults():
    assert config.LLM_MODEL_DEFAULT
    assert config.OLLAMA_HOST_DEFAULT.startswith("http")
    assert 0 < config.LLM_TEMPERATURE <= 2
    assert 0 < config.LLM_TOP_P <= 1
    assert config.LLM_NUM_PREDICT > 0


def test_prompt_template_has_placeholders():
    for placeholder in ("{age_range}", "{gender_focus}", "{word_min}", "{word_max}",
                        "{age_min}", "{age_max}"):
        assert placeholder in config.SYSTEM_PROMPT_TEMPLATE


def test_speed_options_has_default():
    assert config.SPEED_DEFAULT in config.SPEED_OPTIONS
    assert config.SPEED_MIN < config.SPEED_MAX


def test_max_tts_text_length_positive():
    assert config.MAX_TTS_TEXT_LENGTH > 0


def test_story_examples_nonempty():
    assert len(config.STORY_EXAMPLES) >= 3
    assert all(isinstance(x, str) for x in config.STORY_EXAMPLES)


def test_log_settings():
    assert config.LOG_MAX_BYTES > 0
    assert config.LOG_BACKUP_COUNT > 0


def test_gemini_default_model():
    assert "gemini" in config.GEMINI_TTS_MODEL_DEFAULT.lower()
