#!/usr/bin/env python3
"""
Test GPTQ model loader (requires auto-gptq package)
"""
import os
import traceback

def test_gptq_loader(model_id="TheBloke/Llama-2-7B-Chat-GPTQ"):
    """Test loading GPTQ model with integer device"""
    print(f"=== Testing GPTQ Loader: {model_id} ===")
    
    try:
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer
        
        # Load tokenizer
        print("Loading tokenizer...")
        tok = AutoTokenizer.from_pretrained(
            model_id, 
            use_fast=True,
            token=os.getenv("HF_TOKEN")
        )
        print("✓ Tokenizer loaded")
        
        # Load GPTQ model with integer device
        print("Loading GPTQ model...")
        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            device=0,                  # integer index, not "cuda:0"
            use_safetensors=True,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )
        print("✓ GPTQ model loaded on GPU 0")
        
        # Test generation
        prompt = "The benefits of GPU inference are"
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        
        print("Testing generation...")
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
        
    except ImportError as e:
        print(f"✗ auto-gptq not available: {e}")
        print("Install with: pip install auto-gptq")
        return False
    except Exception as e:
        print(f"✗ Error: {type(e).__name__} - {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print()
    
    test_gptq_loader()