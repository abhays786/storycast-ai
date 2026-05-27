"""Tests for tts/voices.py."""

from tts.voices import VoiceInfo, default_voice_for, filter_voices


CATALOG = [
    VoiceInfo(id="a", display_name="A", language="en", gender="boy"),
    VoiceInfo(id="b", display_name="B", language="en", gender="girl"),
    VoiceInfo(id="c", display_name="C", language="hi", gender="both"),
    VoiceInfo(id="d", display_name="D", language="en", gender="both"),
]


def test_filter_voices_by_language():
    out = filter_voices(CATALOG, "hi")
    assert [v.id for v in out] == ["c"]


def test_default_voice_for_exact_gender_match():
    v = default_voice_for(CATALOG, "en", "girl")
    assert v.id == "b"


def test_default_voice_for_falls_back_to_both():
    v = default_voice_for(CATALOG, "en", "alien")
    assert v.id == "d"   # "both" voice in English


def test_default_voice_for_falls_back_to_first_in_language():
    # English-only catalog with only "boy" gender — request "girl" with no
    # "both" voice present.
    cat = [
        VoiceInfo(id="x", display_name="X", language="en", gender="boy"),
    ]
    v = default_voice_for(cat, "en", "girl")
    assert v.id == "x"


def test_default_voice_for_no_language_match_returns_first_overall():
    cat = [
        VoiceInfo(id="x", display_name="X", language="en", gender="boy"),
    ]
    v = default_voice_for(cat, "hi", "boy")
    assert v.id == "x"


def test_default_voice_for_empty_catalog_returns_none():
    assert default_voice_for([], "en", "boy") is None
