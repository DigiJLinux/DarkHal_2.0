from typing import Any, Iterator, List, Optional
import os, json
import torch
from llm_runtime.types import UnifiedModel, GenerateConfig
from llm_runtime.device_utils import device_for_gptq

def _inputs_device_from_gptq_device(dev) -> torch.device:
    """
    AutoGPTQ wants: int GPU index | 'cpu' | 'mps' | 'disk'
    But tokenizer tensors need a torch.device:
      - int -> 'cuda:{idx}'
      - 'cpu' -> 'cpu'
      - 'mps' -> 'mps'
      - 'disk' -> still run the forward on CUDA/CPU; safest default: 'cpu'
    """
    if isinstance(dev, int):
        return torch.device(f"cuda:{dev}")
    if isinstance(dev, str):
        if dev in ("cpu", "mps"):
            return torch.device(dev)
        if dev == "disk":
            # inputs live on CPU; model will page as needed
            return torch.device("cpu")
    # Fallback
    return torch.device("cpu")

class _GPTQUnified:
    def __init__(self, src: str, **kwargs: Any):
        print(f"[GPTQ_DEBUG] _GPTQUnified.__init__() called with src='{src}', kwargs={kwargs}")
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
            print("[GPTQ_DEBUG] Successfully imported auto_gptq and transformers")
        except ImportError as e:
            print(f"[GPTQ_DEBUG] Failed to import auto_gptq or transformers: {e}")
            raise ImportError("auto-gptq and transformers are required for GPTQ models. Install with: pip install auto-gptq transformers")

        print(f"[GPTQ_DEBUG] Loading GPTQ model from: {src}")

        # Normalize device for AutoGPTQ (int or 'cpu'/'mps'/'disk')
        raw_device = kwargs.get("device")
        print(f"[GPTQ_DEBUG] Raw device from kwargs: {raw_device}")
        self._gptq_dev = device_for_gptq(raw_device)
        print(f"[GPTQ_DEBUG] Normalized device for GPTQ: {self._gptq_dev}")
        self._inputs_device = _inputs_device_from_gptq_device(self._gptq_dev)
        print(f"[GPTQ_DEBUG] Using device (gptq): {self._gptq_dev} | inputs will go to: {self._inputs_device}")

        trust_remote = kwargs.get("trust_remote_code", True)
        token = kwargs.get("token")

        # Tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            src,
            use_fast=True,
            trust_remote_code=trust_remote,
            token=token
        )

        # Model
        self.model = AutoGPTQForCausalLM.from_quantized(
            src,
            device=self._gptq_dev,          # int or 'cpu'/'mps'/'disk'
            trust_remote_code=trust_remote,
            use_safetensors=True,
            use_triton=kwargs.get("use_triton", False),
            token=token
        )

        # Pad token safety
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def _build_eos_ids(self, stop) -> Optional[List[int]]:
        """Encode stop strings to token IDs (take the first token of each stop string)."""
        if not stop:
            return None
        out: List[int] = []
        for s in stop:
            if not s:
                continue
            ids = self.tok.encode(s, add_special_tokens=False)
            if ids:
                out.append(ids[0])
        return out or None

    def generate(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> str:
        from transformers import StoppingCriteria, StoppingCriteriaList

        enc = self.tok(prompt, return_tensors="pt").to(self._inputs_device)

        class MultiStringStop(StoppingCriteria):
            def __init__(self, toks, stops):
                self.toks, self.stops = toks, stops or []
            def __call__(self, input_ids, scores, **_):
                # Simple but effective; for high perf, implement a token-level matcher.
                text = self.toks.decode(input_ids[0], skip_special_tokens=True)
                return any(s in text for s in self.stops)

        do_sample = cfg.temperature is not None and cfg.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=cfg.max_tokens,
            do_sample=do_sample,
            top_p=cfg.top_p,
            eos_token_id=self._build_eos_ids(cfg.stop),
            pad_token_id=self.tok.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(cfg.temperature)

        if cfg.stop:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([MultiStringStop(self.tok, cfg.stop)])

        out_ids = self.model.generate(**enc, **gen_kwargs)

        # Decode only new tokens
        new_tokens = out_ids[0][enc.input_ids.shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True)

    def stream(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> Iterator[str]:
        import threading
        from transformers import TextIteratorStreamer

        enc = self.tok(prompt, return_tensors="pt").to(self._inputs_device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

        do_sample = cfg.temperature is not None and cfg.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=cfg.max_tokens,
            do_sample=do_sample,
            top_p=cfg.top_p,
            eos_token_id=self._build_eos_ids(cfg.stop),
            streamer=streamer,
            pad_token_id=self.tok.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(cfg.temperature)

        def _worker():
            self.model.generate(**enc, **gen_kwargs)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        for chunk in streamer:
            yield chunk

    def tokenize(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def detokenize(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

class AutoGPTQLoader:
    name = "gptq"

    def can_load(self, source: str, **kwargs: Any) -> bool:
        # Local folder?
        if os.path.isdir(source):
            # quantize_config.json is a strong GPTQ signal
            qc = os.path.join(source, "quantize_config.json")
            if os.path.exists(qc):
                return True
            # Fallback: peek at config.json
            try:
                with open(os.path.join(source, "config.json"), "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                text = json.dumps(cfg).lower()
                return ("gptq" in text) or ("quantize_config" in text) or ("quant_method" in text)
            except Exception:
                pass
        # HF repo path (heuristic): let the loader try
        elif "/" in source and not os.path.exists(source):
            return True
        return False

    def load(self, source: str, **kwargs: Any) -> UnifiedModel:
        return _GPTQUnified(source, **kwargs)