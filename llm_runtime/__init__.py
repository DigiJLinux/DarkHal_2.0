from .registry import load_model as _load_model
from .types import UnifiedModel, GenerateConfig

# Announce the loaded model to the global spy so other parts of the app can access it
def load_model(*args, **kwargs):
    """
    Proxy to the real registry.load_model that also announces the loaded model via __spy.
    Accepts arbitrary args/kwargs to remain compatible with all loaders.
    """
    model = _load_model(*args, **kwargs)
    try:
        import __spy as spy  # local import to avoid hard dependency during tooling
        # Best-effort extraction of a model name from common argument patterns
        model_name = (
            kwargs.get("source")
            or kwargs.get("model")
            or (args[0] if args else None)
            or getattr(model, "name", None)
            or "unknown"
        )

        # Shallow capture of load parameters (omit non-serializable)
        safe_params = {}
        for k, v in kwargs.items():
            try:
                repr(v)  # ensure it is representable
                safe_params[k] = v
            except Exception:
                continue

        spy.set_model(str(model_name), model, **safe_params)
    except Exception:
        # Never let announcing break model loading
        pass
    return model

__all__ = ["load_model", "UnifiedModel", "GenerateConfig"]