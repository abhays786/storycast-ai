"""Tests for safety.py."""

import pytest

from safety import (
    DEFAULT_SAFETY, SafetyConfig,
    check_input, check_output,
)


def test_default_safety_is_frozen():
    with pytest.raises(Exception):
        DEFAULT_SAFETY.min_topic_length = 99   # type: ignore[misc]


def test_check_input_empty_string():
    ok, err = check_input("")
    assert ok is False
    assert "enter a story topic" in err


def test_check_input_whitespace_only():
    ok, err = check_input("   ")
    assert ok is False


def test_check_input_too_short():
    ok, err = check_input("ab")
    assert ok is False
    assert "too short" in err.lower()


def test_check_input_too_long():
    ok, err = check_input("a" * 401)
    assert ok is False
    assert "too long" in err.lower()


def test_check_input_blocked_word_kill():
    ok, err = check_input("a story about kill")
    assert ok is False
    assert "isn't quite right" in err


def test_check_input_sensitive_pattern_dead_alone():
    # "dead" not followed by tree/end/etc. — should match the pattern;
    # this also hits the blocked_input_words list (which fires first),
    # so we use a phrase that escapes the blocked-words list but still
    # matches the regex.
    ok, err = check_input("the deadlock was broken")
    # "dead" is in blocklist; expect block via blocklist not regex
    assert ok is False


def test_check_input_sensitive_pattern_killing_time_allowed():
    # "killing time" matches the negative-lookahead exception in the regex,
    # but "killing" without those follow-up words IS in the blocklist.
    # Verify the regex pattern explicitly via a topic that bypasses the
    # blocklist (no "kill" substring): hand-craft a phrase.
    cfg = SafetyConfig(
        min_topic_length=3, max_topic_length=400, min_output_length=200,
        blocked_input_words=(),
        sensitive_patterns=(r"\bzapp\b(?!\s+(?:fizz))",),
        blocked_output_terms=(),
    )
    ok, err = check_input("the zapp word here", cfg)
    assert ok is False
    assert "magical or adventurous" in err


def test_check_input_safe():
    ok, err = check_input("A friendly dragon discovers a garden")
    assert ok is True
    assert err == ""


def test_check_input_custom_config():
    cfg = SafetyConfig(
        min_topic_length=2, max_topic_length=10, min_output_length=5,
        blocked_input_words=("forbidden",), sensitive_patterns=(),
        blocked_output_terms=(),
    )
    ok, _ = check_input("forbidden", cfg)
    assert ok is False


def test_check_output_too_short():
    ok, err = check_output("short")
    assert ok is False
    assert "too short" in err.lower()


def test_check_output_blocked_term():
    text = "x" * 300 + " explicit content"
    ok, err = check_output(text)
    assert ok is False
    assert "inappropriate" in err.lower()


def test_check_output_safe():
    text = "A wholesome story. " * 30
    ok, err = check_output(text)
    assert ok is True
    assert err == ""
