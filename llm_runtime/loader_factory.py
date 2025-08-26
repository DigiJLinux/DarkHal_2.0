# loader_factory.py
from typing import Any, Tuple
from .model_router import detect_loader_type, LoaderKind

def select_loader(source: str) -> Tuple[LoaderKind, str]:
    return detect_loader_type(source)

def load_model_for_gui(source: str, **kwargs: Any):
    kind, reason = detect_loader_type(source)
    print(f"[ROUTER] Using {kind.upper()} loader: {reason}")
    print(f"[ROUTER] Source: {source}")
    print(f"[ROUTER] Kwargs: {kwargs}")

    if kind == "hf":
        from .loaders.transformers_loader import HFTransformersLoader
        loader = HFTransformersLoader()
        print(f"[ROUTER] Using HF loader for: {source}")
    else:  # "gguf"
        from .loaders.llamacpp_loader import LlamaCppLoader
        loader = LlamaCppLoader()
        print(f"[ROUTER] Using GGUF loader for: {source}")

    # Load the model
    model = loader.load(source, **kwargs)
    return model, kind, reason