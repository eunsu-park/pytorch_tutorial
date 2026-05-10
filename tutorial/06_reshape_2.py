# 06_reshape_2.py
# reshape(view) 사용 시 -1 활용법
# -1을 사용하면 나머지 차원을 자동으로 계산함을 의미
# 매핑: x.reshape(..., -1) ↔ t.view(..., -1) (PyTorch에서는 reshape도 동일하게 -1 지원)

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── NumPy ──────────────
print("[NumPy]")

array = np.random.randn(2, 2)
print(array)
print(array.shape)
print("")

array = array.reshape(1, -1)              # 앞 차원을 1로 고정, 뒤 차원은 자동 계산 → (1, 4)
print(array)
print(array.shape)
print("")

array = array.reshape(-1, 1)              # 뒤 차원을 1로 고정, 앞 차원은 자동 계산 → (4, 1)
print(array)
print(array.shape)
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

tensor = torch.randn(2, 2)
print(tensor)
print(tensor.shape)
print("")

tensor = tensor.view(1, -1)               # → (1, 4)
print(tensor)
print(tensor.shape)
print("")

tensor = tensor.view(-1, 1)               # → (4, 1)
print(tensor)
print(tensor.shape)
print("")

# ────────────── 비교 정리 ──────────────
# - -1 의 의미는 양쪽 동일: "나머지 차원으로부터 자동 계산"
# - -1은 한 번만 사용 가능 (두 개 이상이면 모호하므로 에러)
# - 자주 쓰는 패턴
#     · 평탄화(flatten) : x.reshape(-1) / t.view(-1)
#     · 배치 유지       : x.reshape(batch, -1) → (batch, 나머지 전부)
