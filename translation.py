"""
Translation between Hindi and English using the local Ollama LLM.

Keeping translation offline mirrors the project's "local-first" stance for
story generation. The translation model is configurable via
`TRANSLATION_MODEL` in .env (defaults to STORY_MODEL).
"""

from app_logging import get_logger
from settings import settings

log = get_logger(__name__)


try:
    import ollama  # type: ignore
    _OLLAMA_RESPONSE_ERROR: type[BaseException] = getattr(ollama, "ResponseError", Exception)
except ImportError:   # pragma: no cover — ollama is a hard dependency
    ollama = None     # type: ignore
    _OLLAMA_RESPONSE_ERROR = Exception


_LANG_NAME = {"en": "English", "hi": "Hindi"}


def _normalize_lang(lang: str | None) -> str:
    """Map Google STT codes like 'hi-IN'/'en-IN' to the ISO 'hi'/'en'."""
    if not lang:
        return "en"
    code = lang.lower().split("-")[0]
    return code if code in _LANG_NAME else "en"


def translate(text: str, source: str | None, target: str) -> tuple[str, str]:
    """
    Translate `text` from `source` language to `target` language.

    Returns `(translated_text, error_message)`. On a same-language no-op or
    when translation is unavailable, returns the original text with an empty
    error and a warning is logged.
    """
    src = _normalize_lang(source)
    tgt = _normalize_lang(target)

    if not text.strip():
        return text, ""
    if src == tgt:
        return text, ""

    if ollama is None:
        return text, "Translation unavailable: Ollama client not installed."

    src_name = _LANG_NAME[src]
    tgt_name = _LANG_NAME[tgt]
    prompt = (
        f"Translate the following text from {src_name} to {tgt_name}. "
        f"Output ONLY the translation, no preface, no explanation. "
        f"Preserve names of people and places.\n\n"
        f"Text:\n{text}"
    )

    model = settings.translation_model
    host  = settings.ollama_host

    log.info("translate: %s → %s model=%s chars=%d", src, tgt, model, len(text))

    try:
        client   = ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system",
                 "content": f"You are a careful translator from {src_name} to {tgt_name}."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2, "top_p": 0.9, "num_predict": 1024},
        )
    except _OLLAMA_RESPONSE_ERROR as exc:
        log.error("Translation Ollama response error: %s", exc)
        return text, f"Translation failed: {exc}"
    except Exception as exc:
        log.error("Translation error: %s", exc)
        return text, f"Translation failed: {exc}"

    out = response["message"]["content"].strip()
    return out or text, ""
