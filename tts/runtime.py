"""
ONNX inference runtime — generic OpenVINO acceleration shim.

Any ONNX-based backend can share this wrapper. Sessions are cached per
ONNX file path so the compile cost is paid once.
"""

from app_logging import get_logger
from tts.devices import ov_device

log = get_logger(__name__)


class OpenVINOSession:
    """Drop-in replacement for onnxruntime.InferenceSession backed by OpenVINO."""

    def __init__(self, compiled_model):
        self._cm = compiled_model

    def run(self, output_names, input_feed, run_options=None):
        result = self._cm(input_feed)
        return [result[o] for o in self._cm.outputs]


_ov_sessions: dict[str, OpenVINOSession] = {}


def get_ov_session(
    onnx_path: str,
    dummy_feed: dict | None = None,
) -> "OpenVINOSession | None":
    """
    Compile or fetch a cached OpenVINO session for the given ONNX file on the
    detected accelerator. Returns None if compilation fails.
    """
    if onnx_path in _ov_sessions:
        return _ov_sessions[onnx_path]

    device = ov_device()
    try:
        import openvino as ov
        core = ov.Core()
        log.info("Compiling model for OpenVINO %s (one-time, ~4s)...", device)
        ov_model = core.read_model(onnx_path)
        compiled = core.compile_model(ov_model, device)
        if dummy_feed is not None:
            compiled(dummy_feed)
        session = OpenVINOSession(compiled)
        _ov_sessions[onnx_path] = session
        log.info("Model ready on OpenVINO %s.", device)
        return session
    except Exception as exc:
        log.warning("OpenVINO compile failed (%s), using CPU ORT.", exc)
        return None


def reset_cache() -> None:
    """Clear the cached compiled sessions (useful for tests)."""
    _ov_sessions.clear()
