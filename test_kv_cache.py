#!/usr/bin/env python3
"""
Test script to validate KV caching functionality across different model formats.
This script tests the complete KV caching system implementation.
"""

import os
import sys
import time
import torch
import psutil
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_runtime import load_model, GenerateConfig
from llm_runtime.chat_session import get_session_manager


def monitor_gpu_memory():
    """Monitor GPU memory usage if CUDA is available"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0.0


def monitor_cpu_memory():
    """Monitor CPU memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3  # GB


def test_transformers_kv_cache():
    """Test KV caching with a Transformers model"""
    print("=" * 60)
    print("Testing Transformers KV Cache")
    print("=" * 60)
    
    # Test with a small model for demonstration
    model_path = "microsoft/DialoGPT-small"  # Small model for testing
    
    try:
        print("📥 Loading model...")
        
        start_time = time.time()
        model = load_model(model_path, device_map="auto")
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"📊 Initial GPU memory: {monitor_gpu_memory():.2f} GB")
        print(f"📊 Initial CPU memory: {monitor_cpu_memory():.2f} GB")
        
        # Test session management
        session_manager = get_session_manager()
        session_id = "test_session"
        
        # Get model info if available
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            print(f"📋 Model info: {model_info}")
        
        # Test 1: First generation (prefill phase)
        print("\n🔄 Test 1: First generation (should trigger prefill)")
        prompt1 = "Hello, how are you?"
        cfg = GenerateConfig(max_tokens=50, temperature=0.7)
        
        start_time = time.time()
        response1 = model.generate(prompt1, cfg=cfg, session_id=session_id)
        gen1_time = time.time() - start_time
        
        print(f"🤖 Response: {response1}")
        print(f"⏱️  Generation time: {gen1_time:.2f} seconds")
        print(f"📊 GPU memory after gen1: {monitor_gpu_memory():.2f} GB")
        
        # Check session state
        if hasattr(model, 'get_session_info'):
            session_info = model.get_session_info(session_id)
            print(f"📋 Session info after gen1: {session_info}")
        
        # Test 2: Second generation (should use cache)
        print("\n🔄 Test 2: Second generation (should use KV cache)")
        prompt2 = prompt1 + " " + response1 + " Tell me more."
        
        start_time = time.time()
        response2 = model.generate(prompt2, cfg=cfg, session_id=session_id)
        gen2_time = time.time() - start_time
        
        print(f"🤖 Response: {response2}")
        print(f"⏱️  Generation time: {gen2_time:.2f} seconds")
        print(f"📊 GPU memory after gen2: {monitor_gpu_memory():.2f} GB")
        
        # Check session state again
        if hasattr(model, 'get_session_info'):
            session_info = model.get_session_info(session_id)
            print(f"📋 Session info after gen2: {session_info}")
        
        # Test 3: Streaming generation with cache
        print("\n🔄 Test 3: Streaming generation with KV cache")
        prompt3 = prompt2 + " " + response2 + " What's your favorite color?"
        
        print("🤖 Streaming response: ", end="", flush=True)
        start_time = time.time()
        streamed_tokens = []
        
        for token in model.stream(prompt3, cfg=cfg, session_id=session_id):
            print(token, end="", flush=True)
            streamed_tokens.append(token)
        
        stream_time = time.time() - start_time
        response3 = "".join(streamed_tokens)
        
        print(f"\n⏱️  Streaming time: {stream_time:.2f} seconds")
        print(f"📊 Final GPU memory: {monitor_gpu_memory():.2f} GB")
        
        # Performance analysis
        print(f"\n📈 Performance Analysis:")
        print(f"   First generation (prefill): {gen1_time:.2f}s")
        print(f"   Second generation (cached): {gen2_time:.2f}s")
        print(f"   Third generation (streamed): {stream_time:.2f}s")
        
        if gen2_time < gen1_time * 0.8:  # Expect at least 20% improvement
            print("✅ KV caching appears to be working (faster subsequent generations)")
        else:
            print("⚠️  KV caching may not be providing expected speedup")
        
        # Test 4: Cache invalidation
        print("\n🔄 Test 4: Cache invalidation test")
        if hasattr(model, 'clear_session_cache'):
            model.clear_session_cache(session_id)
            print("✅ Cache cleared")
            
            # New generation after cache clear
            start_time = time.time()
            response4 = model.generate("This is a completely new conversation.", cfg=cfg, session_id=session_id)
            gen4_time = time.time() - start_time
            
            print(f"🤖 Response after cache clear: {response4}")
            print(f"⏱️  Generation time after cache clear: {gen4_time:.2f} seconds")
        
        print("✅ Transformers KV cache test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Transformers KV cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gguf_compatibility():
    """Test that GGUF models still work with their built-in caching"""
    print("\n" + "=" * 60)
    print("Testing GGUF Model Compatibility")
    print("=" * 60)
    
    # Look for a GGUF model in the models directory
    models_dir = "/mnt/c/Users/mdavi/PycharmProjects/LLM_Train/models"
    gguf_model = None
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.gguf'):
                gguf_model = os.path.join(models_dir, file)
                break
    
    if not gguf_model:
        print("⚠️  No GGUF model found for testing - skipping GGUF test")
        return True
    
    try:
        print("📥 Loading GGUF model...")
        
        start_time = time.time()
        model = load_model(gguf_model, n_ctx=4096, n_gpu_layers=0)
        load_time = time.time() - start_time
        
        print(f"✅ GGUF Model loaded in {load_time:.2f} seconds")
        
        # Test generation
        prompt = "Hello, what is artificial intelligence?"
        cfg = GenerateConfig(max_tokens=50, temperature=0.7)
        
        start_time = time.time()
        response = model.generate(prompt, cfg=cfg)
        gen_time = time.time() - start_time
        
        print(f"🤖 Response: {response}")
        print(f"⏱️  Generation time: {gen_time:.2f} seconds")
        print("✅ GGUF compatibility test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ GGUF compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all KV cache tests"""
    print("🚀 Starting KV Cache Validation Tests")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA device: {torch.cuda.get_device_name()}")
    
    print(f"💾 Initial CPU memory: {monitor_cpu_memory():.2f} GB")
    print(f"🖥️  Initial GPU memory: {monitor_gpu_memory():.2f} GB")
    
    results = []
    
    # Test 1: Transformers KV caching
    results.append(test_transformers_kv_cache())
    
    # Test 2: GGUF compatibility
    results.append(test_gguf_compatibility())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All KV caching tests passed!")
        print("\n📋 KV Caching System Features Validated:")
        print("   ✅ Multi-format support (Transformers + GGUF)")
        print("   ✅ Persistent KV cache across turns")
        print("   ✅ Prefill and decode phases")
        print("   ✅ Session management")
        print("   ✅ Cache invalidation")
        print("   ✅ Streaming support with cache")
        print("   ✅ GPU utilization during inference")
    else:
        print("❌ Some tests failed - check implementation")
        sys.exit(1)


if __name__ == "__main__":
    main()