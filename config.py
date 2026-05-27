"""
Static configuration — constants only.

Runtime / env-derived values live in `settings.py`.
Per-backend voice maps live alongside each backend in `tts/`.
Content guardrail lists and limits live in `safety.py`.
"""

from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
MODEL_DIR         = BASE_DIR / "models"
ASSETS_DIR        = BASE_DIR / "assets"
PREVIEW_CACHE_DIR = ASSETS_DIR / "previews"
LOG_DIR           = BASE_DIR / "logs"

# ── Story generation ──────────────────────────────────────────────────────────
AGE_MIN = 7
AGE_MAX = 14

STORY_WORD_MIN = 200
STORY_WORD_MAX = 1000
STORY_TITLE_MAX_CHARS = 50

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
LLM_MODEL_DEFAULT   = "llama3"
OLLAMA_HOST_DEFAULT = "http://localhost:11434"

LLM_TEMPERATURE = 0.85
LLM_TOP_P       = 0.9
LLM_NUM_PREDICT = 1024

# ── Languages ────────────────────────────────────────────────────────────────
LANGUAGE_OPTIONS = [("English", "en"), ("Hindi", "hi")]
LANGUAGE_LABELS  = {"en": "English", "hi": "Hindi"}
LANGUAGE_DEFAULT = "en"

# ── System prompt template ────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
You are an enthusiastic, warm storyteller who creates short adventures for children.

AUDIENCE: {gender_focus} aged {age_range} years old.

LANGUAGE: {language_instruction}

STORY REQUIREMENTS:
- Length: {word_min}-{word_max} words (reads aloud in roughly 2-5 minutes at a natural pace)
- Structure: vivid opening hook, exciting middle with a challenge, satisfying upbeat resolution
- Tone: lively, imaginative, playful, like a favourite bedtime story
- Language: clear, age-appropriate words; short snappy sentences mixed with longer descriptive ones
- Include: at least two named characters, dialogue, sensory details, a positive moral or takeaway
- Vocabulary: challenging but accessible for a {age_min}-{age_max} year old
- Setting: Use Indian context for names, mountains, places

ABSOLUTE CONTENT RULES (no exceptions):
- No violence, gore, graphic peril, or lasting harm to characters
- No adult, romantic, or sexual themes
- No references to drugs, alcohol, tobacco, or harmful substances
- No real-world disasters, wars, or traumatic events
- No bullying celebrated or left unaddressed
- No scary content beyond light fun adventure tension
- No crude language or insults

OUTPUT FORMAT - respond with EXACTLY this structure, nothing else:
Title: <a short, vivid title under {title_max_chars} characters>

<story text paragraphs>
"""

LANGUAGE_INSTRUCTIONS = {
    "en": "Write the entire response in clear, natural English.",
    "hi": "पूरा उत्तर सरल, स्वाभाविक हिंदी (देवनागरी लिपि) में लिखें। शीर्षक भी हिंदी में हो।",
}

# ── TTS general ───────────────────────────────────────────────────────────────
SPEED_MIN     = 0.5
SPEED_MAX     = 2.0
SPEED_OPTIONS = ["0.7x", "0.8x", "0.9x", "1.0x (Normal)", "1.2x", "1.3x"]
SPEED_DEFAULT = "1.0x (Normal)"

# Playback-rate dropdown shown next to the rendered audio (browser-side only).
PLAYBACK_RATE_OPTIONS = ["0.7x", "0.8x", "0.9x", "1.0x", "1.1x", "1.2x"]
PLAYBACK_RATE_DEFAULT = "1.0x"

MAX_TTS_TEXT_LENGTH = 8000

# ── Voice preview ─────────────────────────────────────────────────────────────
PREVIEW_TEXT = {
    "en": "Hello! I'll be the voice of your stories. I can speak gently, brightly, and "
          "with plenty of expression. Pick me to give your tales a friendly sound.",
    "hi": "नमस्ते! मैं आपकी कहानियों की आवाज़ बनूँगा। मैं प्यार से, स्पष्ट और भावपूर्ण ढंग से बोल सकता हूँ। "
          "मुझे चुनें और अपनी कहानियों को एक मधुर रूप दें।",
}

# ── Gemini TTS model default ──────────────────────────────────────────────────
GEMINI_TTS_MODEL_DEFAULT = "gemini-2.5-flash-preview-tts"

# ── UI examples ───────────────────────────────────────────────────────────────
STORY_EXAMPLES = [
    "A dragon who is afraid of fire but wants to be brave",
    "A robot who learns to dance at the school talent show",
    "A girl who discovers a secret door in the library",
    "A boy who finds a magical backpack that grants one wish a day",
    "A tiny dinosaur who wants to play with the big kids",
    "A friendly cloud who learns why rain is a gift",
]

# ── Logging ───────────────────────────────────────────────────────────────────
APP_LOG_DIR      = LOG_DIR / "app"
LOG_MAX_BYTES    = 15 * 1024 * 1024   # 15 MB per file before rolling
LOG_BACKUP_COUNT = 30                  # keep 30 rolled files
