# 04_numpy_vs_pytorch.py
# numpy array와 pytorch tensor의 생성·dtype 차이
# 매핑: np.array([...])     ↔ torch.tensor([...])
#       np.array(..., dtype=np.int64)   ↔ torch.LongTensor([...]) / torch.tensor(..., dtype=torch.int64)
#       np.array(..., dtype=np.float32) ↔ torch.FloatTensor([...]) / torch.tensor(..., dtype=torch.float32)

import numpy as np
import torch

# ────────────── NumPy ──────────────
print("[NumPy]")

# rank-1, 정수 (기본 dtype = int64)
array = np.array([1, 2, 3])
print(array.shape, array.dtype)

# rank-1, 실수 (기본 dtype = float64)
array = np.array([1., 2., 3.])
print(array.shape, array.dtype)

# rank-2, 정수
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array.shape, array.dtype)

# dtype 명시 - int64
array = np.array([1, 2, 3], dtype=np.int64)
print(array.shape, array.dtype)

# dtype 명시 - float32
array = np.array([1, 2, 3], dtype=np.float32)
print(array.shape, array.dtype)
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

# rank-1, 정수 (기본 dtype = torch.int64)
tensor = torch.tensor([1, 2, 3])              # torch.tensor() : torch tensor 생성 함수
print(tensor.shape, tensor.dtype)

# rank-1, 실수 (기본 dtype = torch.float32, NumPy의 float64와 다름)
tensor = torch.tensor([1., 2., 3.])
print(tensor.shape, tensor.dtype)

# rank-2, 정수
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor.shape, tensor.dtype)

# dtype 명시 - int64 (LongTensor)
tensor = torch.LongTensor([1, 2, 3])          # torch.LongTensor() : int64 tensor 생성
print(tensor.shape, tensor.dtype)

# dtype 명시 - float32 (FloatTensor)
tensor = torch.FloatTensor([1, 2, 3])         # torch.FloatTensor() : float32 tensor 생성
print(tensor.shape, tensor.dtype)
print("")

# ────────────── 비교 정리 ──────────────
# - 생성 함수
#     NumPy   : np.array([...])
#     PyTorch : torch.tensor([...])           ← 권장
#               torch.LongTensor / torch.FloatTensor : 타입별 단축 생성자(레거시 스타일)
# - 기본 dtype
#     정수 : NumPy=int64    ↔ PyTorch=torch.int64    (사실상 동일)
#     실수 : NumPy=float64  ↔ PyTorch=torch.float32  ★중요: 다름
# - shape 표시
#     NumPy   : tuple   예) (3,)
#     PyTorch : torch.Size  예) torch.Size([3])  → tuple 처럼 인덱싱 가능
