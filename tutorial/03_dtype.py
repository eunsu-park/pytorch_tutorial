# 03_dtype.py
# 데이터 타입(dtype) 확인 및 지정
# 매핑: x.dtype ↔ t.dtype  /  np.int32/np.float32 ↔ torch.int32/torch.float32

import numpy as np
import torch

# ────────────── NumPy ──────────────
print("[NumPy]")

# 정수형 데이터
x = np.array([1, 2, 3, 4, 5])
print(f"Data Type: {x.dtype}")           # 기본 정수 dtype: int64 (플랫폼에 따라 다를 수 있음)

# 정수형 데이터 (dtype 지정)
x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
print(f"Data Type: {x.dtype}")

# 실수형 데이터
x = np.array([1., 2., 3., 4., 5.])
print(f"Data Type: {x.dtype}")           # 기본 실수 dtype: float64

# 실수형 데이터 (dtype 지정)
x = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
print(f"Data Type: {x.dtype}")
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

# 정수형 데이터
t = torch.tensor([1, 2, 3, 4, 5])
print(f"Data Type: {t.dtype}")           # 기본 정수 dtype: torch.int64

# 정수형 데이터 (dtype 지정)
t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
print(f"Data Type: {t.dtype}")

# 실수형 데이터
t = torch.tensor([1., 2., 3., 4., 5.])
print(f"Data Type: {t.dtype}")           # 기본 실수 dtype: torch.float32 (NumPy와 다름)

# 실수형 데이터 (dtype 지정)
t = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
print(f"Data Type: {t.dtype}")
print("")

# ────────────── 비교 정리 ──────────────
# - dtype 확인     : x.dtype ↔ t.dtype
# - 정수 기본형    : NumPy=int64    ↔  PyTorch=torch.int64       (사실상 동일)
# - 실수 기본형    : NumPy=float64  ↔  PyTorch=torch.float32     ★중요: 다름
#     · PyTorch가 float32를 기본으로 쓰는 이유는 GPU 연산 효율 때문
#     · 따라서 NumPy → PyTorch 변환 시 float64가 그대로 넘어와 학습이 느려질 수 있으니 주의
# - dtype 지정     : dtype=np.int32 ↔ dtype=torch.int32 (네임스페이스만 다름)
