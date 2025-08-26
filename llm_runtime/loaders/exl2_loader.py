from typing import Any, Iterator, List, Optional
import os
from llm_runtime.types import UnifiedModel, GenerateConfig

class _ExLlama2Unified:
    def __init__(self, src: str, **kwargs: Any):
        try:
            from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, ExLlamaV2Cache
            from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
        except ImportError:
            raise ImportError("exllamav2 is required for EXL2 models. Install with: pip install exllamav2")
        
        print(f"Loading EXL2 model from: {src}")
        
        # Configure model
        self.config = ExLlamaV2Config(src)
        
        # Apply any config overrides
        if "max_seq_len" in kwargs:
            self.config.max_seq_len = kwargs["max_seq_len"]
        if "scale_pos_emb" in kwargs:
            self.config.scale_pos_emb = kwargs["scale_pos_emb"]
        if "scale_alpha_value" in kwargs:
            self.config.scale_alpha_value = kwargs["scale_alpha_value"]
        
        # Initialize model
        self.model = ExLlamaV2(self.config)
        
        # Load model weights
        self.model.load()
        
        # Initialize tokenizer
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        
        # Initialize cache
        self.cache = ExLlamaV2Cache(self.model, lazy=kwargs.get("lazy_cache", True))
        
        # Initialize generator for streaming
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        
        print(f"EXL2 model loaded successfully")

    def generate(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> str:
        # Create sampler settings
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = cfg.temperature
        settings.top_p = cfg.top_p
        
        # Set stop conditions
        if cfg.stop:
            # ExLlamaV2 expects stop strings as a list
            stop_conditions = list(cfg.stop)
        else:
            stop_conditions = []
        
        # Generate text
        output = self.generator.generate_simple(
            prompt=prompt,
            max_new_tokens=cfg.max_tokens,
            seed=kwargs.get("seed", -1),
            token_healing=kwargs.get("token_healing", True),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop_conditions=stop_conditions,
        )
        
        return output

    def stream(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> Iterator[str]:
        # Create sampler settings
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = cfg.temperature
        settings.top_p = cfg.top_p
        
        # Set stop conditions
        if cfg.stop:
            stop_conditions = list(cfg.stop)
        else:
            stop_conditions = []
        
        # Begin streaming generation
        input_ids = self.tokenizer.encode(prompt)
        self.generator.begin_stream(
            input_ids=input_ids,
            gen_settings=settings,
            token_healing=kwargs.get("token_healing", True),
            seed=kwargs.get("seed", -1),
        )
        
        generated_tokens = 0
        
        while generated_tokens < cfg.max_tokens:
            chunk, eos, tokens = self.generator.stream()
            
            if chunk:
                yield chunk
            
            generated_tokens += tokens
            
            # Check for stop conditions
            if eos:
                break
            
            if cfg.stop:
                # Check if any stop condition is met in the generated text so far
                current_text = self.generator.sequence_str
                if any(stop in current_text for stop in cfg.stop):
                    break

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).tolist()

    def detokenize(self, ids: List[int]) -> str:
        import torch
        tensor_ids = torch.tensor([ids], dtype=torch.long)
        return self.tokenizer.decode(tensor_ids)[0]

class ExLlama2Loader:
    name = "exllama2"
    
    def can_load(self, source: str, **kwargs: Any) -> bool:
        # Check for EXL2 model directory structure
        if os.path.isdir(source):
            # Look for config.json and .safetensors files that indicate EXL2
            config_path = os.path.join(source, "config.json")
            if os.path.exists(config_path):
                # Check if there are .safetensors files with EXL2 naming pattern
                for file in os.listdir(source):
                    if file.endswith(".safetensors") and ("model" in file.lower() or "exl2" in file.lower()):
                        return True
        elif source.lower().endswith(".exl2"):
            return True
        
        return False
    
    def load(self, source: str, **kwargs: Any) -> UnifiedModel:
        return _ExLlama2Unified(source, **kwargs)