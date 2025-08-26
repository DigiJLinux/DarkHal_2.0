#!/usr/bin/env python3
"""
Quick sanity check for PyTorch and CUDA setup
"""
import torch

def check_torch_setup():
    """Check PyTorch and CUDA configuration"""
    print("=== PyTorch & CUDA Debug Info ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda runtime:", torch.version.cuda)
    
    if torch.cuda.is_available():
        print("gpu count:", torch.cuda.device_count(), "name:", torch.cuda.get_device_name(0))
        print("cuda arch list:", torch.cuda.get_arch_list())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, Memory: {props.total_memory/1024**3:.1f}GB")
    else:
        print("WARNING: CUDA not available - you may have CPU-only torch")
        print("To fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Test basic tensor operations
    try:
        x = torch.randn(2, 2)
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print("✓ Basic CUDA tensor operations work")
        print("✓ Basic CPU tensor operations work")
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")

if __name__ == "__main__":
    check_torch_setup()