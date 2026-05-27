# StoryCast AI — KidsStory Magic

A local-first Gradio app that turns a short topic (typed or spoken) into a
kid-friendly story and reads it aloud in a chosen voice.

---

## Executive summary

**What it does.** A child (or parent) picks a topic, an audience focus
(boy / girl / both), and a TTS engine. The app drafts a 200–1000 word adventure
in 2–5 minutes of read-aloud time using a local Llama 3 model (no story text
ever leaves the machine), then synthesises the narration to a WAV using one of
several swappable backends.

**Audience.** Stories are tuned for ages 7–14, with content guardrails on both
the input topic and the generated text (no violence, romance, substances, or
real-world trauma).

**Why local.** The story generation runs entirely on the user's machine via
[Ollama](https://ollama.com), so prompts and generated stories stay private.
Speech synthesis can also be 100 % offline (Piper); Gemini Flash TTS is offered
as an opt-in cloud alternative for higher-quality voices.

**Hardware acceleration.** On Intel Arc GPUs / NPUs, Piper inference is
automatically dispatched through OpenVINO. BARK and Coqui pick up CUDA or
Intel XPU when available, with CPU fallback everywhere.

**Two UI themes.** A compact `classic` card layout and a `modern`
purple/green two-panel layout, switchable via the `UI_THEME` env var.

---

## Architecture

```
storycast-ai/
├── app.py              entry point — wires settings, theme, registry, launch
├── settings.py         single env-resolution composition root
├── config.py           static constants (paths, prompts, age range, examples)
├── safety.py           injectable input/output content guardrails
├── agent.py            story generation via local Llama 3 / Ollama
├── pipeline.py         orchestration — yields semantic Event sum type
├── stt.py              speech-to-text (Google STT via SpeechRecognition)
├── session_archive.py  per-session artifact persistence under logs/
├── app_logging.py      stdlib logging with daily + size-based rotation
├── tts_engine.py       dispatcher with central fallback policy
├── tts/
│   ├── base.py         TTSBackend ABC (id, display_name, generate, ...)
│   ├── registry.py     register / lookup / filter
│   ├── bootstrap.py    register_default_backends() called by app.py
│   ├── devices.py      lazy memoized device detection (torch / openvino)
│   ├── runtime.py      shared OpenVINO session shim
│   ├── _utils.py       speed adjustment, text chunking, output-path helper
│   ├── piper_backend.py
│   ├── gemini_backend.py
│   ├── coqui_backend.py
│   └── bark_backend.py
└── ui/
    ├── themes.py       Theme dataclass + CLASSIC / MODERN definitions
    ├── widgets.py      banner, audio_settings, parse_speed_label
    └── builder.py      single themed builder — emits BuiltApp(blocks, css)
```

### Key design decisions

- **TTS backends are a plugin protocol.** `TTSBackend` (`id`, `display_name`,
  `generate`, `is_available`, `ensure_models`) lets new engines drop in
  without changes elsewhere. Registration is explicit
  (`register_default_backends()`), never an import side-effect.
- **Pipeline emits semantic events, not Gradio tuples.** `pipeline.py` yields a
  `Union[Transcribing, Transcribed, GeneratingStory, StoryGenerated,
  Synthesizing, AudioReady, Failed]`. The UI adapter in `ui/builder.py`
  matches on type and produces the Gradio output tuple. A future CLI or REST
  front-end can reuse the same orchestration.
- **Themes are data, not code.** `Theme` holds explicit class lists, CSS,
  titles, and structural toggles. The builder never branches on `theme.name`,
  so adding a third theme = adding a `Theme` instance.
- **Settings live in one place.** Every env-derived value is parsed in
  `settings.py`; `load_dotenv()` is called exactly once. No other module reads
  `os.getenv`.
- **Guardrails are injectable.** `safety.SafetyConfig` lets callers swap
  blocklists / thresholds without monkey-patching the module.

---

## Setup

### Prerequisites

- **Python 3.10+** (development uses 3.14)
- **Ollama** running locally — install from <https://ollama.com/download>,
  then `ollama pull llama3`
- **(Optional) Gemini API key** if you want the `gemini` TTS engine

### Install

```powershell
git clone <repo-url> storycast-ai
cd storycast-ai
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux / macOS replace the activate line with `source .venv/bin/activate`.

### Configure (`.env`)

Create a `.env` file in the repo root. Every variable is optional — defaults
are shown.

```dotenv
# UI
UI_THEME=classic                  # classic | modern
ENABLED_BACKENDS=piper,gemini     # comma-separated; backends shown in the UI

# Local LLM (Ollama)
STORY_MODEL=llama3
OLLAMA_HOST=http://localhost:11434

# Cloud TTS (only needed if you enable the Gemini engine)
GEMINI_API_KEY=
GEMINI_TTS_MODEL=gemini-2.5-flash-preview-tts
```

The `.env` file is ignored by git (see `.gitignore`).

---

## Running the app

```powershell
py app.py
```

Opens <http://localhost:7860> in your default browser. The first launch will
download Piper voice models (~50 MB total) into `models/` from HuggingFace.

### What happens at startup

1. `tts/bootstrap.register_default_backends()` registers all four backends.
2. `settings.py` loads `.env`; `check_ollama()` probes the configured host
   and logs a hint if the model isn't pulled (non-fatal).
3. `ensure_all_models()` walks every registered backend's `ensure_models()`
   hook (Piper downloads, others are no-ops).
4. Gradio Blocks are built for the chosen theme and `launch()`-ed.

### Tabs

- **Story Magic** — type or speak a topic; the LLM writes a story and the
  chosen TTS engine reads it.
- **Text to Audio** — paste up to 8000 characters of arbitrary text and
  convert it directly to speech.

---

## Development

### Project layout in one paragraph

`app.py` is the entry point only — it wires settings, registers backends,
builds the UI, and launches Gradio. `pipeline.py` owns orchestration and is
front-end agnostic. `ui/` consumes the pipeline. `tts/` is a self-contained
plugin layer; everything else (`agent`, `stt`, `safety`, `session_archive`,
`app_logging`) is a single-purpose module.

### Adding a new TTS backend

1. Create `tts/<name>_backend.py`:

   ```python
   from tts.base import TTSBackend
   from tts._utils import new_output_path

   class MyBackend(TTSBackend):
       id = "myengine"

       @property
       def display_name(self) -> str:
           return "My Engine — Friendly Name"

       def is_available(self) -> bool:
           return True   # check env / library presence here

       def generate(self, text: str, gender: str, speed: float) -> str | None:
           out = new_output_path()
           # ... synthesise into `out` ...
           return str(out) if out.exists() else None
   ```

2. Register it in `tts/bootstrap.py` and add `"myengine"` to your `.env`
   `ENABLED_BACKENDS` list to expose it in the dropdown.

3. Add a test file `tests/test_tts_myengine.py` covering happy + failure
   paths (mock the external library via `sys.modules` injection — see
   `tests/test_tts_gemini.py` for the pattern).

### Adding a new theme

1. Add a CSS string and a `Theme(...)` instance in `ui/themes.py`.
2. Extend `get_theme()` to recognise its name.
3. No changes to `ui/builder.py` should be required — the builder reads every
   structural decision off the `Theme` dataclass.

### Tweaking content safety

Pass a custom `SafetyConfig` to `check_input` / `check_output`:

```python
from safety import SafetyConfig, check_input

my_policy = SafetyConfig(
    min_topic_length=5, max_topic_length=200, min_output_length=300,
    blocked_input_words=("foo", "bar"),
    sensitive_patterns=(),
    blocked_output_terms=(),
)
ok, err = check_input("a story", my_policy)
```

For agent-internal guardrails the default policy is used — change it in
`safety.DEFAULT_SAFETY` or refactor `agent.generate_story` to take a config.

---

## Tests

```powershell
py run_tests.py
```

Runs the full pytest suite with coverage. The script fails (non-zero exit) if
either tests fail or total coverage drops below 100 %.

```
241 tests, 1023 statements, 100 % coverage in ~10 s
```

Forwarded args work too:

```powershell
py run_tests.py tests/test_pipeline.py        # one file
py run_tests.py -k "speed" -v                 # filter + verbose
py run_tests.py --cov-report=html             # HTML coverage report
```

### How the suite stays hermetic

Heavy external dependencies are mocked via `sys.modules` injection:
`ollama`, `piper`, `google.genai`, `TTS.api`, `bark`, `speech_recognition`,
`openvino`, `pydub`, `huggingface_hub`. No real network calls, no model
downloads, no Gradio server.

`tests/conftest.py` redirects `ASSETS_DIR`, `LOG_DIR`, and `MODEL_DIR` into
`tmp_path` per test, replaces every project logger's handlers with
`NullHandler`, and clears the TTS registry between tests.

---

## Maintenance

### File locations

| Path                          | Purpose                                              |
|-------------------------------|------------------------------------------------------|
| `logs/app/kidsstory.log`      | Stdlib app log (15 MB × 30 rolled files)             |
| `logs/inputs/text/`           | Per-session topic + TTS input text                   |
| `logs/inputs/audio/`          | Mic recordings copied from Gradio's temp files       |
| `logs/outputs/text/`          | Generated story title + body                         |
| `logs/outputs/audio/`         | Synthesised WAVs                                     |
| `assets/`                     | Most-recent synthesis output (timestamped WAVs)      |
| `models/`                     | Downloaded Piper voices (`.onnx` + `.onnx.json`)     |

Everything under `logs/`, `assets/*.wav`, and `models/*.onnx*` is git-ignored.

### Common troubleshooting

- **"Ollama does not appear to be running"** — start the Ollama daemon and
  run `ollama pull llama3`. The app continues to run; only story generation
  is blocked.
- **"Model '…' is not pulled yet"** — `ollama pull <STORY_MODEL>` and restart.
- **Gemini option missing from the dropdown** — set `GEMINI_API_KEY` in
  `.env`; the backend's `is_available()` returns `False` without it.
- **Coqui / BARK missing from dropdown** — they ship disabled. Add their
  ids to `ENABLED_BACKENDS` in `.env` and ensure the underlying libraries
  install cleanly (`coqui-tts`, `suno-bark` in `requirements.txt`).
- **Piper synthesis crashes on Intel hardware** — drop OpenVINO acceleration
  by uninstalling `openvino`. The backend falls back to CPU ONNX Runtime
  automatically.
- **Stale model files** — delete `models/<voice-id>.onnx*` to force a
  redownload on next launch.

### Releasing / pinning

`requirements.txt` is the canonical install manifest. When upgrading a
backend library, run the full test suite and the smoke launch in both
themes:

```powershell
py run_tests.py
py app.py     # ctrl-C after the UI loads in both classic and modern
```

### Reproducible runs from a clean checkout

```powershell
git clean -fdx                 # nuke logs, assets, models, .venv
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py run_tests.py                # confirm 100% pass + coverage
py app.py
```

---

## License & attribution

- Story generation: Meta Llama 3 via Ollama
- Offline TTS: [Piper](https://github.com/rhasspy/piper) voices from
  `rhasspy/piper-voices` on HuggingFace
- Cloud TTS: Google Gemini Flash TTS
- UI: [Gradio](https://gradio.app)
- Optional: Coqui TTS (jenny model), Suno BARK
