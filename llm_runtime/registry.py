from typing import Any
from .types import UnifiedModel
from .loader_factory import load_model_for_gui

def load_model(source: str, **kwargs: Any) -> UnifiedModel:
    """Load model using the router-based factory system"""
    print(f"[REGISTRY_DEBUG] load_model() called with source='{source}', kwargs={kwargs}")
    model, kind, reason = load_model_for_gui(source, **kwargs)
    print(f"[REGISTRY_DEBUG] load_model_for_gui() returned: kind='{kind}', reason='{reason}'")
    return model