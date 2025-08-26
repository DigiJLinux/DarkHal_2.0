from typing import Any, Iterator, List, Optional
import os
from llm_runtime.types import UnifiedModel, GenerateConfig

class _AWQUnified:
    def __init__(self, src: str, **kwargs: Any):
        try:
            from autoawq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("autoawq and transformers are required for AWQ models. Install with: pip install autoawq transformers")
        
        print(f"Loading AWQ model from: {src}")
        
        # Load tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            src, 
            use_fast=True,
            trust_remote_code=kwargs.get("trust_remote_code", True)
        )
        
        # Load AWQ model
        self.model = AutoAWQForCausalLM.from_quantized(
            src,
            device_map=kwargs.get("device_map", "auto"),
            trust_remote_code=kwargs.get("trust_remote_code", True),
            safetensors=True,
        )
        
        # Set pad token if not present
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def _build_eos_ids(self, stop) -> Optional[List[int]]:
        """Convert stop strings to token IDs"""
        if not stop: 
            return None
        
        ids = []
        for s in stop:
            if len(s) == 1:
                tid = self.tok.convert_tokens_to_ids(s)
                if tid is not None: 
                    ids.append(tid)
        return ids or None

    def generate(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> str:
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        class MultiStringStop(StoppingCriteria):
            def __init__(self, toks, stops): 
                self.toks, self.stops = toks, stops or []
            
            def __call__(self, input_ids, scores, **_):
                text = self.toks.decode(input_ids[0], skip_special_tokens=True)
                return any(s in text for s in self.stops)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_tokens,
            do_sample=cfg.temperature > 0.0,
            temperature=cfg.temperature if cfg.temperature > 0.0 else None,
            top_p=cfg.top_p,
            eos_token_id=self._build_eos_ids(cfg.stop),
            stopping_criteria=StoppingCriteriaList([MultiStringStop(self.tok, cfg.stop)]) if cfg.stop else None,
            pad_token_id=self.tok.pad_token_id,
        )
        
        # Decode only the new tokens
        generated_text = self.tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text

    def stream(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> Iterator[str]:
        import threading
        from transformers import TextIteratorStreamer
        
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        
        def _worker():
            self.model.generate(
                **enc,
                max_new_tokens=cfg.max_tokens,
                do_sample=cfg.temperature > 0.0,
                temperature=cfg.temperature if cfg.temperature > 0.0 else None,
                top_p=cfg.top_p,
                eos_token_id=self._build_eos_ids(cfg.stop),
                streamer=streamer,
                pad_token_id=self.tok.pad_token_id,
            )
        
        t = threading.Thread(target=_worker)
        t.start()
        
        for text in streamer:
            yield text

    def tokenize(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def detokenize(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

class AWQLoader:
    name = "awq"
    
    def can_load(self, source: str, **kwargs: Any) -> bool:
        # Check for AWQ indicators
        if os.path.isdir(source):
            # Look for AWQ indicators in config.json
            try:
                import json
                config_path = os.path.join(source, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        config_str = str(config).lower()
                        return ("awq" in config_str or 
                               "quantization_config" in config and 
                               "awq" in str(config.get("quantization_config", {})).lower())
            except:
                pass
        elif "/" in source and not os.path.exists(source):  # HF repo
            # For HF repos, we'll let it try if it looks like AWQ
            return "awq" in source.lower()
        
        return False
    
    def load(self, source: str, **kwargs: Any) -> UnifiedModel:
        return _AWQUnified(source, **kwargs)