# 07_unsqueeze.py
# 차원 추가(unsqueeze/expand_dims) 및 제거(squeeze)
# 매핑: np.expand_dims(x, axis) ↔ torch.unsqueeze(t, dim) / t.unsqueeze(dim)
#       np.squeeze(x, axis)     ↔ torch.squeeze(t, dim)   / t.squeeze(dim)

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── NumPy ──────────────
print("[NumPy]")

x = np.random.rand(32, 32)
y = np.expand_dims(x, axis=0)             # 0번째 위치에 차원 추가 → (1, 32, 32)
print(y.shape)
print("")

z = np.squeeze(y, axis=0)                 # 0번째 차원(크기 1) 제거 → (32, 32)
print(z.shape)
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

x = torch.randn(32, 32)
y = torch.unsqueeze(x, 0)                 # 0번째 위치에 차원 추가, x.unsqueeze(0) 도 동일
print(y.shape)
print("")

z = torch.squeeze(y, 0)                   # 0번째 차원(크기 1) 제거, y.squeeze(0) 도 동일
print(z.shape)
print("")

# ────────────── 비교 정리 ──────────────
# - 차원 추가
#     NumPy   : np.expand_dims(x, axis=k)
#     PyTorch : torch.unsqueeze(t, k) 또는 t.unsqueeze(k)
# - 차원 제거
#     NumPy   : np.squeeze(x, axis=k)        (k 생략 시 크기 1인 차원 모두 제거)
#     PyTorch : torch.squeeze(t, k)          (k 생략 시 동일)
# - 자주 쓰는 패턴
#     · 단일 이미지 → 배치 차원 추가 : (C, H, W) → (1, C, H, W)
#     · 배치 결과의 채널 차원 제거   : (N, 1, H, W) → (N, H, W)
