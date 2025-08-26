from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Protocol, Any

@dataclass
class GenerateConfig:
    # None means "use model defaults"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Iterable[str]] = None
    # Optional context window hint; models may ignore if they manage it internally
    context_window: Optional[int] = None
    # Optional advanced knobs; models may ignore if unsupported
    min_p: Optional[float] = None
    typical_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

class UnifiedModel(Protocol):
    # Default config uses None for all tunables so the model can choose
    def generate(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> str: ...
    def stream(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> Iterator[str]: ...
    def tokenize(self, text: str) -> List[int]: ...
    def detokenize(self, ids: List[int]) -> str: ...