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
print(f"rank-0  : ndim={x.ndim}, shape={x.shape}    | dim={t.dim()}, shape={t.shape}")

# rank-1 (벡터)
x = np.array([1, 2, 3, 4, 5])
t = torch.tensor([1, 2, 3, 4, 5])
print(f"rank-1  : ndim={x.ndim}, shape={x.shape}  | dim={t.dim()}, shape={t.shape}")

# rank-2 (행렬)
x = np.zeros((3, 5))
t = torch.zeros(3, 5)
print(f"rank-2  : ndim={x.ndim}, shape={x.shape}  | dim={t.dim()}, shape={t.shape}")

# rank-3 (예: 영상 (C, H, W))
x = np.zeros((3, 32, 32))
t = torch.zeros(3, 32, 32)
print(f"rank-3  : ndim={x.ndim}, shape={x.shape}  | dim={t.dim()}, shape={t.shape}")
print("")

# ────────────── (2) 자주 쓰는 생성 함수 ──────────────
# 신경망에서 가중치 초기화·예시 데이터 생성에 자주 사용됨.
print("[2] 텐서 생성 함수")

print("zeros    :", torch.zeros(2, 3))                    # 모두 0
print("ones     :", torch.ones(2, 3))                     # 모두 1
print("full     :", torch.full((2, 3), 7.0))              # 모두 같은 값
print("eye      :", torch.eye(3))                          # 단위행렬
print("arange   :", torch.arange(0, 10, 2))                # [0, 10) step 2
print("linspace :", torch.linspace(0, 1, 5))               # 0~1 사이 5개
print("randn    :", torch.randn(2, 3))                     # 표준정규
print("rand     :", torch.rand(2, 3))                      # [0, 1) 균일분포
print("")

# 같은 shape 의 새 텐서를 만들 때 유용한 *_like 함수
ref = torch.randn(2, 3)
print("zeros_like(ref) :", torch.zeros_like(ref).shape)
print("ones_like(ref)  :", torch.ones_like(ref).shape)
print("")

# ────────────── (3) 숫자 한 개 꺼내기 ──────────────
# rank-0 tensor 를 파이썬 숫자로 바꾸려면 .item()
t = torch.tensor(3.14)
print(f"t.item() = {t.item()}  (type = {type(t.item()).__name__})")

# ────────────── 비교 정리 ──────────────
# - 차원 수    : np.ndarray.ndim   ↔  torch.Tensor.dim()    (속성 vs 메서드)
# - 차원 크기  : np.ndarray.shape  ↔  torch.Tensor.shape    (혹은 .size())
# - 생성 함수
#     NumPy   : np.zeros / np.ones / np.full / np.eye / np.arange / np.linspace / np.random.{randn, rand}
#     PyTorch : torch.zeros / torch.ones / torch.full / torch.eye / torch.arange / torch.linspace / torch.{randn, rand}
# - 학습용 가짜 데이터는 보통 torch.randn(batch, channel, H, W) 형태로 생성
