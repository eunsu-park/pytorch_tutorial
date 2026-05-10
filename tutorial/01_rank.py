# 01_rank.py
# 데이터의 차원(rank)과 크기(shape) 확인
# 매핑: x.ndim ↔ t.dim()  /  x.shape ↔ t.shape (혹은 t.size())

import numpy as np
import torch

# ────────────── NumPy ──────────────
print("[NumPy]")

print("Rank-0 Tensor")
x = np.array(1)              # np.array() : numpy array 생성 함수
print(f"Rank: {x.ndim}")     # x.ndim  : 차원 수
print(f"Shape: {x.shape}")   # x.shape : 차원의 크기
print(x)
print("")

print("Rank-1 Tensor")
x = np.array([1, 2, 3, 4, 5])
print(f"Rank: {x.ndim}")
print(f"Shape: {x.shape}")
print(x[0])                   # 0번째 원소
print("")

print("Rank-2 Tensor")
x = np.array([[11, 12, 13, 14, 15],
              [21, 22, 23, 24, 25],
              [31, 32, 33, 34, 35]])
print(f"Rank: {x.ndim}")
print(f"Shape: {x.shape}")
print(x[0])                   # 0번째 행
print(x[0, -1])               # 0번째 행의 마지막 열
print(x[0][-1])               # 0번째 행의 마지막 열, 위와 동일한 결과
print("")

print("Rank-3 Tensor")
x = np.array([[[101, 102, 103, 104, 105],
               [111, 112, 113, 114, 115],
               [121, 122, 123, 124, 125]],
              [[201, 202, 203, 204, 205],
               [211, 212, 213, 214, 215],
               [221, 222, 223, 224, 225]]])
print(f"Rank: {x.ndim}")
print(f"Shape: {x.shape}")
print(x[0])                   # 0번째 행렬
print(x[0][1])                # 0번째 행렬의 1번째 행
print(x[0][1][2])             # 0번째 행렬의 1번째 행의 2번째 열
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

print("Rank-0 Tensor")
t = torch.tensor(1)           # torch.tensor() : torch tensor 생성 함수
print(f"Rank: {t.dim()}")     # t.dim()  : 차원 수 (NumPy의 ndim에 해당)
print(f"Shape: {t.shape}")    # t.shape : 차원의 크기 (t.size()와 동일)
print(t)
print("")

print("Rank-1 Tensor")
t = torch.tensor([1, 2, 3, 4, 5])
print(f"Rank: {t.dim()}")
print(f"Shape: {t.shape}")
print(t[0])                   # 0번째 원소 (rank-0 tensor로 반환됨)
print("")

print("Rank-2 Tensor")
t = torch.tensor([[11, 12, 13, 14, 15],
                  [21, 22, 23, 24, 25],
                  [31, 32, 33, 34, 35]])
print(f"Rank: {t.dim()}")
print(f"Shape: {t.shape}")
print(t[0])                   # 0번째 행
print(t[0, -1])               # 0번째 행의 마지막 열
print(t[0][-1])               # 위와 동일한 결과
print("")

print("Rank-3 Tensor")
t = torch.tensor([[[101, 102, 103, 104, 105],
                   [111, 112, 113, 114, 115],
                   [121, 122, 123, 124, 125]],
                  [[201, 202, 203, 204, 205],
                   [211, 212, 213, 214, 215],
                   [221, 222, 223, 224, 225]]])
print(f"Rank: {t.dim()}")
print(f"Shape: {t.shape}")
print(t[0])                   # 0번째 행렬
print(t[0][1])                # 0번째 행렬의 1번째 행
print(t[0][1][2])             # 0번째 행렬의 1번째 행의 2번째 열
print("")

# ────────────── 비교 정리 ──────────────
# - 차원 수    : np.ndarray.ndim  ↔  torch.Tensor.dim()  (속성 vs 메서드)
# - 차원 크기  : np.ndarray.shape ↔  torch.Tensor.shape  (혹은 .size())
# - 인덱싱 문법(x[0], x[0,-1], x[0][-1]) 은 양쪽 모두 동일하게 동작
# - 단, PyTorch에서 스칼라 하나를 인덱싱하면 rank-0 tensor가 반환되므로,
#   파이썬 숫자가 필요하면 .item() 을 사용
