import torch
from typing import Union, Literal

Backend = Literal["hf", "gptq"]
DevIn = Union[None, str, int]
DevOut = Union[str, int]

def _has_mps() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

def _first_cuda_index() -> int | None:
    return 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None

def normalize_device(dev: DevIn = None, *, backend: Backend = "hf") -> DevOut:
    """
    Normalize a user/device string/int into what each backend expects.

    Inputs accepted:
      None | "auto" | "cpu" | "mps" | "disk" | "cuda" | "cuda:N" | N (int)

    Returns:
      backend == "hf":   "cpu" | "mps" | "cuda:N"
      backend == "gptq": "cpu" | "mps" | "disk" | N (int)
    """
    # 1) Auto/default
    if dev in (None, "auto"):
        cuda0 = _first_cuda_index()
        if cuda0 is not None:
            return (cuda0 if backend == "gptq" else f"cuda:{cuda0}")
        if _has_mps():
            return "mps"
        # AutoGPTQ can also run with 'disk' offload if caller wants; default to CPU here
        return "cpu"

    # 2) Explicit CPU/MPS/DISK
    if isinstance(dev, str) and dev.lower() in {"cpu", "mps", "disk"}:
        # HF does not know "disk"; treat as CPU for HF branch
        return dev if backend == "gptq" or dev != "disk" else "cpu"

    # 3) Explicit CUDA string
    if isinstance(dev, str) and dev.lower().startswith("cuda"):
        # Accept "cuda" and "cuda:N"
        if dev == "cuda":
            idx = _first_cuda_index()
            if idx is None:
                # No CUDA available; degrade to CPU/MPS appropriately
                return "cpu" if backend == "hf" else "cpu"
            return (idx if backend == "gptq" else f"cuda:{idx}")
        # cuda:N
        try:
            idx = int(dev.split(":", 1)[1])
        except (IndexError, ValueError):
            raise ValueError(f"Bad CUDA device string: {dev!r}. Use 'cuda' or 'cuda:N'.")
        return (idx if backend == "gptq" else f"cuda:{idx}")

    # 4) Integer GPU index
    if isinstance(dev, int):
        if dev < 0:
            raise ValueError(f"GPU index must be >= 0, got {dev}")
        return (dev if backend == "gptq" else f"cuda:{dev}")

    raise ValueError(f"Unsupported device spec for backend={backend!r}: {dev!r}")

# --- Convenience wrappers -------------------------------------------------------

def device_for_hf(dev: DevIn = None) -> str:
    """Return a device string suitable for HuggingFace (e.g., 'cuda:0', 'cpu', 'mps')."""
    out = normalize_device(dev, backend="hf")
    assert isinstance(out, str)
    return out

def device_for_gptq(dev: DevIn = None) -> Union[int, str]:
    """Return an int GPU index or 'cpu'/'mps'/'disk' for AutoGPTQ."""
    out = normalize_device(dev, backend="gptq")
    assert isinstance(out, (int, str))
    return out

def debug_device_placement(model, name="model"):
    """Debug helper to check where model parameters are placed"""
    try:
        devices = set()
        for name_param, param in model.named_parameters():
            devices.add(str(param.device))
        print(f"[DEBUG] {name} parameters on devices: {devices}")
        
        # Check first parameter device
        first_param = next(model.parameters())
        print(f"[DEBUG] {name} primary device: {first_param.device}")
        
        return first_param.device
    except Exception as e:
        print(f"[DEBUG] Could not check {name} device placement: {e}")
        return None

# --- Minimal self-test (run this file directly) ---------------------------------
if __name__ == "__main__":
    tests = [None, "auto", "cpu", "mps", "disk", "cuda", "cuda:0", "cuda:1", 0, 1]
    for t in tests:
        print(f"in={t!r:7}  -> hf={device_for_hf(t)!r:7}  gptq={device_for_gptq(t)!r}")