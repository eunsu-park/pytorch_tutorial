# 01_tensor_basics.py
# 텐서 기본 — 차원(rank) / 크기(shape) / 생성 함수
# 매핑: x.ndim ↔ t.dim(),  x.shape ↔ t.shape(또는 t.size())

import numpy as np
import torch

# ────────────── (1) 차원과 크기 ──────────────
print("[1] 차원과 크기")

# rank-0 (스칼라)
x = np.array(1)
t = torch.tensor(1)
print(f"  rank-0 : ndim={x.ndim}, shape={x.shape}    | dim={t.dim()}, shape={t.shape}")

# rank-1 (벡터)
x = np.array([1, 2, 3, 4, 5])
t = torch.tensor([1, 2, 3, 4, 5])
print(f"  rank-1 : ndim={x.ndim}, shape={x.shape}  | dim={t.dim()}, shape={t.shape}")

# rank-2 (행렬)
x = np.zeros((3, 5))
t = torch.zeros(3, 5)
print(f"  rank-2 : ndim={x.ndim}, shape={x.shape}  | dim={t.dim()}, shape={t.shape}")

# rank-3 (예: 영상 (C, H, W))
x = np.zeros((3, 32, 32))
t = torch.zeros(3, 32, 32)
print(f"  rank-3 : ndim={x.ndim}, shape={x.shape}  | dim={t.dim()}, shape={t.shape}")
print("")

# ────────────── (2) 자주 쓰는 생성 함수 — NumPy ──────────────
# NumPy 의 거의 모든 생성 함수는 shape 를 'tuple' 로 받는다.
print("[2-A] NumPy 생성 함수")
print("  np.zeros((2,3))     :\n", np.zeros((2, 3)))
print("  np.ones((2,3))      :\n", np.ones((2, 3)))
print("  np.full((2,3), 7.)  :\n", np.full((2, 3), 7.0))
print("  np.eye(3)           :\n", np.eye(3))
print("  np.arange(0, 10, 2) :", np.arange(0, 10, 2))
print("  np.linspace(0, 1, 5):", np.linspace(0, 1, 5))
print("  np.random.randn(2,3):\n", np.random.randn(2, 3))
print("  np.random.rand(2,3) :\n", np.random.rand(2, 3))
print("")

# ────────────── (2) 자주 쓰는 생성 함수 — PyTorch ──────────────
# PyTorch 는 shape 인자를 'tuple' 또는 '여러 정수 인자' 둘 다 받는다.
#   torch.zeros(2, 3)   ↔   torch.zeros((2, 3))   둘 다 동작
print("[2-B] PyTorch 생성 함수")
print("  torch.zeros(2, 3)        :\n", torch.zeros(2, 3))
print("  torch.ones(2, 3)         :\n", torch.ones(2, 3))
print("  torch.full((2, 3), 7.0)  :\n", torch.full((2, 3), 7.0))
print("  torch.eye(3)             :\n", torch.eye(3))
print("  torch.arange(0, 10, 2)   :", torch.arange(0, 10, 2))
print("  torch.linspace(0, 1, 5)  :", torch.linspace(0, 1, 5))
print("  torch.randn(2, 3)        :\n", torch.randn(2, 3))
print("  torch.rand(2, 3)         :\n", torch.rand(2, 3))
print("")

# ────────────── (3) 같은 shape 의 새 텐서 — *_like ──────────────
ref_np = np.random.rand(2, 3)
ref_pt = torch.randn(2, 3)
print("[3] *_like 함수 (ref 의 shape 따라 생성)")
print(f"  np.zeros_like(ref)        : {np.zeros_like(ref_np).shape}")
print(f"  np.ones_like(ref)         : {np.ones_like(ref_np).shape}")
print(f"  torch.zeros_like(ref)     : {torch.zeros_like(ref_pt).shape}")
print(f"  torch.ones_like(ref)      : {torch.ones_like(ref_pt).shape}")
print("")

# ────────────── (4) 숫자 한 개 꺼내기 ──────────────
# rank-0 tensor/array 를 파이썬 숫자로 꺼낼 때 양쪽 모두 .item() 가능
x = np.array(3.14)
t = torch.tensor(3.14)
print(f"[4] x.item() = {x.item()}    t.item() = {t.item()}    (둘 다 파이썬 float 반환)")

# ────────────── 비교 정리 ──────────────
# 차원 수    : np.ndarray.ndim  ↔  torch.Tensor.dim()      (속성 vs 메서드)
# 차원 크기  : np.ndarray.shape ↔  torch.Tensor.shape       (혹은 .size())
# 생성 함수
#   영행렬·일행렬 :  np.zeros / np.ones                 ↔  torch.zeros / torch.ones
#   고정값        :  np.full(shape, value)              ↔  torch.full(shape, value)
#   단위행렬      :  np.eye(n)                          ↔  torch.eye(n)
#   등간격        :  np.arange / np.linspace            ↔  torch.arange / torch.linspace
#   정규/균일난수 :  np.random.randn / np.random.rand   ↔  torch.randn / torch.rand
#   shape 복제    :  np.zeros_like(ref) / np.ones_like  ↔  torch.zeros_like / torch.ones_like
# 인자 차이
#   NumPy 는 shape 를 'tuple' 로만 받음:  np.zeros((2, 3))
#   PyTorch 는 'tuple' 또는 '여러 정수' 둘 다:  torch.zeros(2, 3) / torch.zeros((2, 3))
