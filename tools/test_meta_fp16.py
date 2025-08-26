#!/usr/bin/env python3
"""
Minimal loader test for Meta's FP16/BF16 Llama models (no GPTQ)
"""
import os
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_meta_fp16_loader():
    """Test loading Meta's Llama model with device_map=auto"""
    MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"  # Meta's repo
    
    print(f"=== Testing Meta FP16 Loader: {MODEL_ID} ===")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            use_fast=True,
            token=os.getenv("HF_TOKEN")
        )
        print("✓ Tokenizer loaded")
        
        # Load model with device_map="auto"
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",          # let accelerate place it
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )
        
        # Check device placement
        device = next(model.parameters()).device
        print(f"✓ Model loaded on device: {device}")
        
        # Test generation
        prompt = "The benefits of GPU inference are"
        inputs = tok(prompt, return_tensors="pt").to(device)
        
        print("Testing generation...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        
        result = tok.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation test passed:")
        print(f"Output: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__} - {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print()
    
    test_meta_fp16_loader()