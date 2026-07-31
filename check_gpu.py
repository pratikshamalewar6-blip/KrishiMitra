import os
import torch

print("=" * 50)
print("          SYSTEM HARDWARE & CUDA CHECKER")
print("=" * 50)

# 1. CPU Check
print(f"💻 CPU Cores Available : {os.cpu_count()}")

# 2. GPU & CUDA Check
cuda_available = torch.cuda.is_available()
print(f"🚀 CUDA Available      : {cuda_available}")

if cuda_available:
    print(f"🔥 GPU Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"⚡ GPU Count           : {torch.cuda.device_count()}")
    
    # VRAM Check
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    print(f"💾 Total VRAM (GPU)    : {vram_gb:.2f} GB")
else:
    print("⚠️ CUDA nahi mila! Model sirf CPU par chalega (jo thoda slow ho sakta hai).")

print("=" * 50)
