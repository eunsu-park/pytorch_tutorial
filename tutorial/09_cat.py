# 09_cat.py
# 차원 기준으로 데이터 합치기 (concatenate / cat)
# 매핑: np.concatenate([x, y], axis=k) ↔ torch.cat([x, y], dim=k)

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── NumPy ──────────────
print("[NumPy]")

x = np.random.rand(1, 32, 32)
y = np.random.rand(1, 32, 32)

z = np.concatenate([x, y], axis=0)        # 0번째 차원을 기준으로 합침 → (2, 32, 32)
print(z.shape)
print("")

w = np.concatenate([x, y], axis=1)        # 1번째 차원을 기준으로 합침 → (1, 64, 32)
print(w.shape)
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

x = torch.randn(1, 32, 32)
y = torch.randn(1, 32, 32)

z = torch.cat([x, y], dim=0)              # 0번째 차원 기준 → (2, 32, 32)
print(z.shape)
print("")

w = torch.cat([x, y], dim=1)              # 1번째 차원 기준 → (1, 64, 32)
print(w.shape)
print("")

# ────────────── 비교 정리 ──────────────
# - 동등 함수
#     NumPy   : np.concatenate([x, y, ...], axis=k)
#     PyTorch : torch.cat([x, y, ...], dim=k)
# - 인자 이름이 axis ↔ dim 으로 다름
# - 합치는 차원을 제외한 나머지 차원의 크기는 동일해야 함 (양쪽 모두)
# - 새로운 차원을 만들면서 쌓고 싶다면
#     NumPy   : np.stack([x, y], axis=k)
#     PyTorch : torch.stack([x, y], dim=k)
