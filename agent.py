"""
Story generation agent using a local LLM via Ollama.

Override defaults via .env (resolved centrally in settings.py):
  STORY_MODEL        — model used for English stories (default: llama3)
  HINDI_STORY_MODEL  — model used for Hindi stories   (default: STORY_MODEL)
  OLLAMA_HOST        — Ollama server URL              (default: localhost:11434)
"""

from app_logging import get_logger
from config import (
    AGE_MAX, AGE_MIN,
    LANGUAGE_INSTRUCTIONS,
    LLM_NUM_PREDICT, LLM_TEMPERATURE, LLM_TOP_P,
    STORY_TITLE_MAX_CHARS, STORY_WORD_MAX, STORY_WORD_MIN,
    SYSTEM_PROMPT_TEMPLATE,
)
from safety import check_input, check_output
from settings import settings

log = get_logger(__name__)


try:
    import ollama  # type: ignore
    _OLLAMA_RESPONSE_ERROR: type[BaseException] = getattr(ollama, "ResponseError", Exception)
except ImportError:   # pragma: no cover — ollama is a hard dependency
    ollama = None     # type: ignore
    _OLLAMA_RESPONSE_ERROR = Exception


GENDER_FOCUS_MAP = {
    "boy":  "boys",
    "girl": "girls",
    "both": "all children (boys and girls equally)",
}


def _select_model(language: str) -> str:
    """Pick the configured Ollama model for a story language."""
    return settings.hindi_story_model if language == "hi" else settings.story_model


def _truncate_title(title: str) -> str:
    """Keep titles compact and safe for filename use."""
    title = (title or "").strip().strip('"').strip("'")
    if len(title) > STORY_TITLE_MAX_CHARS:
        title = title[:STORY_TITLE_MAX_CHARS].rstrip(" .,;:-—")
    return title or "A Wonderful Story"


def generate_story(
    topic: str,
    gender: str,
    age_min: int = AGE_MIN,
    age_max: int = AGE_MAX,
    language: str = "en",
) -> tuple[str, str, str]:
    """
    Generate a kids' story using a local LLM via Ollama.

    Returns:
        (title, story_text, error_message)
        On success: title and story_text are populated, error_message is "".
        On failure: title and story_text are "", error_message describes the issue.
    """
    safe, err = check_input(topic)
    if not safe:
        return "", "", err

    if ollama is None:
        return "", "", "Ollama Python client is not installed."

    age_range    = f"{age_min}-{age_max}"
    gender_focus = GENDER_FOCUS_MAP.get(gender.lower(), "all children")
    lang_instr   = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        age_range=age_range,
        gender_focus=gender_focus,
        word_min=STORY_WORD_MIN,
        word_max=STORY_WORD_MAX,
        age_min=age_min,
        age_max=age_max,
        language_instruction=lang_instr,
        title_max_chars=STORY_TITLE_MAX_CHARS,
    )

    model = _select_model(language)
    host  = settings.ollama_host

    log.info("generate_story: topic=%r gender=%s language=%s model=%s",
             topic, gender, language, model)

    try:
        client   = ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Create a story about: {topic}"},
            ],
            options={
                "temperature": LLM_TEMPERATURE,
                "top_p":       LLM_TOP_P,
                "num_predict": LLM_NUM_PREDICT,
            },
        )

    except _OLLAMA_RESPONSE_ERROR as exc:
        log.error("Ollama response error: %s", exc)
        if "model" in str(exc).lower() and "not found" in str(exc).lower():
            return "", "", (
                f"Model '{model}' is not pulled yet. "
                f"Run:  ollama pull {model}  — then restart the app."
            )
        return "", "", f"Ollama error: {exc}"

    except Exception as exc:
        log.error("Story generation error: %s", exc)
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg or "connect" in msg:
            return "", "", (
                "Cannot reach Ollama. Make sure it is running: "
                "download from https://ollama.com/download and start it, "
                f"then pull the model with:  ollama pull {model}"
            )
        return "", "", f"Unexpected error: {exc}"

    raw = response["message"]["content"].strip()

    title = "A Wonderful Story"
    story_lines_start = 0
    lines = raw.split("\n")

    for i, line in enumerate(lines):
        if line.lower().startswith("title:"):
            title = line[len("title:"):].strip()
            story_lines_start = i + 1
            break

    title = _truncate_title(title)

    while story_lines_start < len(lines) and not lines[story_lines_start].strip():
        story_lines_start += 1

    story_text = "\n".join(lines[story_lines_start:]).strip()

    ok, out_err = check_output(story_text)
    if not ok:
        log.warning("Output guardrail failed: %s", out_err)
        return "", "", out_err

    log.info("Story generated: title=%r words=~%d", title, len(story_text.split()))
    return title, story_text, ""
