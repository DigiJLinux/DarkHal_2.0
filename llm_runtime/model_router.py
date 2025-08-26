from __future__ import annotations
import os, json
from pathlib import Path
from typing import Literal, Optional, Dict, Tuple, Callable

# Only support Hugging Face (Safetensors) and GGUF
LoaderKind = Literal["hf", "gguf"]

def detect_loader_type(source: str) -> Tuple[LoaderKind, str]:
    """
    Decide which loader to use based on the path/repo.
    Returns: (kind, reason)
      kind ∈ {"hf","gguf"}
    """
    print(f"[ROUTER_DEBUG] detect_loader_type() called with source: '{source}'")
    p = Path(source)
    low = source.lower()

    # 0) Force HF for official Meta repos
    if not p.exists() and low.startswith("meta-llama/"):
        result = ("hf", "Official Meta repo (full-precision).")
        print(f"[ROUTER_DEBUG] Meta repo detected -> {result}")
        return result

    # 1) Local FILE
    if p.exists() and p.is_file():
        if p.suffix.lower() == ".gguf":
            return "gguf", "Local .gguf file."
        return "hf", "Local non-.gguf file (default HF)."

    # 2) Local DIR
    if p.exists() and p.is_dir():
        # GGUF hint: any *.gguf inside
        if any(p.glob("*.gguf")):
            return "gguf", "Directory contains .gguf file(s)."
        # Inspect config.json only for GGUF hints; all else defaults to HF
        cfg = p / "config.json"
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                text = json.dumps(data).lower()
                if ("gguf" in text) or ("llama.cpp" in text) or ("ggml" in text):
                    return "gguf", "config.json mentions GGUF/llama.cpp."
            except Exception:
                pass
        return "hf", "Default HF for non-GGUF directories."

    # 3) Remote repo style (org/name)
    if ("/" in source or source.count("\\") == 1) and not p.exists():
        if any(tag in low for tag in ["gguf", "ggml", "llama.cpp"]):
            return "gguf", "Repo name suggests GGUF."
        return "hf", "Remote repo (default HF)."

    # 4) Fallback
    return "hf", "Fallback to HF."