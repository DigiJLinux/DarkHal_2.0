from typing import Any, Iterator, List
from llm_runtime.types import UnifiedModel, GenerateConfig

class _LlamaCppUnified:
    def __init__(self, model_path: str, **kwargs: Any):
        # Import llama_cpp directly instead of from main.py to avoid circular imports
        from llama_cpp import Llama
        
        if not model_path.lower().endswith(".gguf"):
            raise ValueError(f"Not a valid GGUF model: {model_path}")
        
        self.model_path = model_path
        self.kwargs = kwargs
        self._llama = None
        
    def _get_model(self):
        """Lazy load the model using existing implementation"""
        if self._llama is None:
            # Import the _get_llama function from main module to maintain compatibility
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                from main import _get_llama
                self._llama = _get_llama(
                    self.model_path,
                    n_ctx=self.kwargs.get("n_ctx", 8192),
                    n_gpu_layers=self.kwargs.get("n_gpu_layers", 0),
                    lora_path=self.kwargs.get("lora_path"),
                    n_threads=self.kwargs.get("n_threads"),
                )
            except ImportError:
                # Fallback to direct llama-cpp-python if main import fails
                from llama_cpp import Llama
                self._llama = Llama(
                    model_path=self.model_path,
                    n_ctx=self.kwargs.get("n_ctx", 8192),
                    n_gpu_layers=self.kwargs.get("n_gpu_layers", 0),
                    verbose=self.kwargs.get("verbose", False),
                    n_threads=self.kwargs.get("n_threads")
                )
        return self._llama

    def generate(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> str:
        llama = self._get_model()
        
        # Convert GenerateConfig to llama-cpp-python format
        result = llama(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=list(cfg.stop) if cfg.stop else None,
            echo=False
        )
        
        return result["choices"][0]["text"]

    def stream(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> Iterator[str]:
        llama = self._get_model()
        
        # Use streaming generation
        for chunk in llama.create_completion(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=list(cfg.stop) if cfg.stop else None,
            stream=True,
            echo=False
        ):
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    def tokenize(self, text: str) -> List[int]:
        llama = self._get_model()
        return llama.tokenize(text.encode("utf-8"), add_bos=False)

    def detokenize(self, ids: List[int]) -> str:
        llama = self._get_model()
        return llama.detokenize(ids).decode("utf-8", errors="ignore")

class LlamaCppLoader:
    name = "llamacpp"
    
    def can_load(self, source: str, **kwargs: Any) -> bool:
        return source.lower().endswith(".gguf")
    
    def load(self, source: str, **kwargs: Any) -> UnifiedModel:
        return _LlamaCppUnified(source, **kwargs)