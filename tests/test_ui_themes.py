"""Tests for ui/themes.py."""

from ui.themes import CLASSIC, MODERN, Theme, get_theme


def test_get_theme_classic():
    assert get_theme("classic") is CLASSIC


def test_get_theme_modern():
    assert get_theme("modern") is MODERN


def test_get_theme_unknown_defaults_to_classic():
    assert get_theme("dark") is CLASSIC


def test_get_theme_empty_defaults_to_classic():
    assert get_theme("") is CLASSIC


def test_get_theme_none_defaults_to_classic():
    assert get_theme(None) is CLASSIC


def test_get_theme_case_insensitive():
    assert get_theme("MODERN") is MODERN


def test_theme_immutable():
    import pytest
    with pytest.raises(Exception):
        CLASSIC.name = "x"   # type: ignore[misc]


def test_themes_have_unique_panel_classes():
    assert CLASSIC.left_panel_classes != MODERN.left_panel_classes


def test_modern_has_card_titles_classic_doesnt():
    assert CLASSIC.story_card_title is None
    assert MODERN.story_card_title is not None


def test_themes_are_named_consistently():
    assert CLASSIC.name == "classic"
    assert MODERN.name == "modern"
