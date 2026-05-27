"""
Device detection — memoized so importing this module costs nothing;
the first call to each function does the (cached) probe.

Public:
    torch_device() -> "cuda" | "xpu" | "cpu"
    ov_device()    -> "GPU"  | "NPU" | "CPU"
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def torch_device() -> str:
    """cuda > xpu (Intel via IPEX) > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError:
            pass
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def ov_device() -> str:
    """Best OpenVINO device for ONNX inference: GPU > NPU > CPU."""
    try:
        import openvino as ov
        devices = ov.Core().available_devices
        for preferred in ("GPU", "NPU"):
            if preferred in devices:
                return preferred
    except Exception:
        pass
    return "CPU"


def reset_cache() -> None:
    """Clear the memoized detection (used by tests)."""
    torch_device.cache_clear()
    ov_device.cache_clear()
