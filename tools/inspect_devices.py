import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

def dtype_nbytes(dt: torch.dtype) -> int:
    return {
        torch.float32: 4, torch.float: 4,
        torch.float16: 2, torch.bfloat16: 2,
        torch.int8: 1, torch.uint8: 1,
        torch.int4: 0.5,  # pseudo for 4-bit quant libs
    }.get(dt, 4)

def pretty_bytes(n: float) -> str:
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024 or u == "TB": return f"{n:.2f} {u}"
        n /= 1024

def inspect_model_devices(model_path_or_id: str) -> str:
    """Inspect where model parameters are placed and return detailed report"""
    output = []
    
    try:
        output.append(f"=== Inspecting Model: {model_path_or_id} ===\n")
        
        # Load model as-is (don't force a map yet—show reality)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id, 
            torch_dtype="auto", 
            device_map="auto", 
            low_cpu_mem_usage=True
        )
        
        output.append(f">>> hf_device_map present: {hasattr(model, 'hf_device_map')}")
        if hasattr(model, "hf_device_map"):
            output.append(">>> device_map (first 20 entries):")
            for i, (k, v) in enumerate(model.hf_device_map.items()):
                if i < 20: 
                    output.append(f"  {k:40s} -> {v}")
            if len(model.hf_device_map) > 20: 
                output.append(f"  ... and {len(model.hf_device_map) - 20} more entries")
        
        totals = {}
        by_dtype = {}
        on_meta = []
        
        for n, p in model.named_parameters():
            dev = str(p.device)
            totals[dev] = totals.get(dev, 0) + p.numel() * p.element_size()
            by_dtype[p.dtype] = by_dtype.get(p.dtype, 0) + p.numel() * p.element_size()
            if dev == "meta":
                on_meta.append(n)
        
        output.append("\n=== Bytes by device ===")
        for dev, b in totals.items():
            output.append(f"  {dev:10s} : {pretty_bytes(b)}")
        
        output.append("\n=== Bytes by dtype ===")
        for dt, b in by_dtype.items():
            output.append(f"  {str(dt):12s} : {pretty_bytes(b)}")
        
        if on_meta:
            output.append(f"\n⚠️  WARNING: {len(on_meta)} parameters on META (not really loaded). Examples:")
            for n in on_meta[:10]:
                output.append(f"    - {n}")
            if len(on_meta) > 10:
                output.append(f"    ... and {len(on_meta) - 10} more")
        
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            output.append(f"\n=== CUDA Memory ===")
            output.append(f"  Used: {pretty_bytes(used)} / Total: {pretty_bytes(total)} on cuda:0")
            output.append(f"  Free: {pretty_bytes(free)} ({(free/total)*100:.1f}%)")
        else:
            output.append("\n❌ CUDA not available.")
        
        # Quick check if fully on GPU
        all_cuda = all(str(p.device).startswith("cuda") for _, p in model.named_parameters())
        no_meta = not any(str(p.device) == "meta" for _, p in model.named_parameters())
        
        output.append(f"\n=== Summary ===")
        if all_cuda and no_meta:
            output.append("✅ All parameters are on CUDA")
        else:
            output.append("❌ Model is NOT fully on GPU")
            if on_meta:
                output.append("   - Some parameters are on META device")
            if not all_cuda:
                output.append("   - Some parameters are on CPU")
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        output.append(f"❌ Error inspecting model: {str(e)}")
    
    return "\n".join(output)

def inspect_loaded_model(model) -> str:
    """Inspect an already loaded model"""
    output = []
    
    try:
        output.append("=== Inspecting Currently Loaded Model ===\n")
        
        totals = {}
        by_dtype = {}
        on_meta = []
        
        for n, p in model.named_parameters():
            dev = str(p.device)
            totals[dev] = totals.get(dev, 0) + p.numel() * p.element_size()
            by_dtype[p.dtype] = by_dtype.get(p.dtype, 0) + p.numel() * p.element_size()
            if dev == "meta":
                on_meta.append(n)
        
        output.append("=== Bytes by device ===")
        for dev, b in totals.items():
            output.append(f"  {dev:10s} : {pretty_bytes(b)}")
        
        output.append("\n=== Bytes by dtype ===")
        for dt, b in by_dtype.items():
            output.append(f"  {str(dt):12s} : {pretty_bytes(b)}")
        
        if on_meta:
            output.append(f"\n⚠️  WARNING: {len(on_meta)} parameters on META. Examples:")
            for n in on_meta[:5]:
                output.append(f"    - {n}")
        
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            output.append(f"\n=== CUDA Memory ===")
            output.append(f"  Used: {pretty_bytes(used)} / Total: {pretty_bytes(total)}")
            output.append(f"  Free: {pretty_bytes(free)} ({(free/total)*100:.1f}%)")
        
        # Quick check
        all_cuda = all(str(p.device).startswith("cuda") for _, p in model.named_parameters())
        no_meta = not any(str(p.device) == "meta" for _, p in model.named_parameters())
        
        output.append(f"\n=== Summary ===")
        if all_cuda and no_meta:
            output.append("✅ All parameters are on CUDA")
        else:
            output.append("❌ Model is NOT fully on GPU")
        
    except Exception as e:
        output.append(f"❌ Error: {str(e)}")
    
    return "\n".join(output)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        result = inspect_model_devices(model_path)
        print(result)
    else:
        print("Usage: python inspect_devices.py <model_path_or_id>")