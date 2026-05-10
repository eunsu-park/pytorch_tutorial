# 08_permute.py
# 차원 순서 변경 (transpose/permute)
# 매핑: np.transpose(x, axes) / x.transpose(*axes) ↔ torch.permute(t, dims) / t.permute(*dims)

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── NumPy ──────────────
print("[NumPy]")

x = np.random.rand(32, 32, 3)             # (H, W, C) — 영상 처리에서 흔한 NumPy 관습
print(x.shape)
print("")

y = np.transpose(x, (2, 0, 1))            # (H, W, C) → (C, H, W) — PyTorch 관습으로 변경
# (2, 0, 1) : 새 0축 = 원래 2축, 새 1축 = 원래 0축, 새 2축 = 원래 1축
print(y.shape)
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

x = torch.randn(32, 32, 3)
print(x.shape)
print("")

y = torch.permute(x, (2, 0, 1))           # x.permute(2, 0, 1) 도 동일
print(y.shape)
print("")

# ────────────── 비교 정리 ──────────────
# - 모든 차원 재배치
#     NumPy   : np.transpose(x, (...))           ※ 모든 차원 가능
#     PyTorch : torch.permute(t, (...))          ※ 모든 차원 가능
# - 두 차원 교환
#     NumPy   : np.swapaxes(x, a, b)             또는 np.transpose(x, ...)
#     PyTorch : torch.transpose(t, a, b)         ※ PyTorch의 transpose는 두 차원 교환만 가능 (NumPy와 의미 다름)
# - 영상 데이터 관습 차이
#     NumPy/OpenCV/Matplotlib : (H, W, C)
#     PyTorch                 : (C, H, W)        → 학습 전에 permute(2, 0, 1) 자주 사용
