# === FIX PUSH ke baad — fresh run ===
!rm -rf yupp
!git clone -b main https://github.com/ashutoshdadhich999-ops/yupp.git
%cd yupp

!pip install -q torch torchvision torchaudio snntorch numpy matplotlib seaborn tqdm

import torch
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

# Quick test (same as pehle)
!python main.py --epochs-img 3 --epochs-audio 3 --ablation-epochs 2 --audio-subset-size 2000

# Jab quick pass ho jaye, full run:
# !python main.py

print("\nFigures:")
!ls -la outputs/figures/
