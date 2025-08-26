#!/usr/bin/env python3
"""
__spy.py
Lightweight global announcer for the currently loaded model.

Usage:
- Call set_model(model_name, model_obj, **params) when a model is loaded.
- Retrieve with get_model(), get_model_name(), or get_info() anywhere.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import threading

@dataclass
class SpyData:
    model_name: str
    model: Any
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Avoid serializing the raw model object
        d["model"] = repr(self.model)
        return d

_lock = threading.RLock()
_current: Optional[SpyData] = None

def set_model(model_name: str, model: Any, **params: Any) -> None:
    """Announce the current model and its load parameters."""
    global _current
    with _lock:
        _current = SpyData(model_name=model_name, model=model, params=dict(params or {}))

def get_model() -> Optional[Any]:
    """Return the current model object, if any."""
    with _lock:
        return _current.model if _current else None

def get_model_name() -> Optional[str]:
    """Return the current model name, if any."""
    with _lock:
        return _current.model_name if _current else None

def get_info() -> Optional[SpyData]:
    """Return the full SpyData object, if any."""
    with _lock:
        return _current
