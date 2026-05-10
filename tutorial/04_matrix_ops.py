# 04_matrix_ops.py
# 행렬 연산과 broadcasting — element-wise / dot / matmul / broadcasting
#
# 학습 목표
#   회귀(16~18) 와 Linear 레이어(19~20) 에서 곧바로 쓰이는 핵심 연산을 미리 정리한다.
#   특히 broadcasting 과 '* 는 matmul 이 아니다' 같은 흔한 함정을 학습자가 직접 본다.

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) Element-wise 연산 ──────────────
# 같은 shape 끼리 +, -, *, /, ** 등은 위치별로 계산. 양쪽 동일하게 동작.
print("[1] Element-wise")

a_np = np.array([[1., 2., 3.], [4., 5., 6.]])
b_np = np.array([[10., 10., 10.], [20., 20., 20.]])

a_pt = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
b_pt = torch.tensor([[10., 10., 10.], [20., 20., 20.]])

print("a + b (NumPy)\n", a_np + b_np)
print("a + b (PyTorch)\n", a_pt + b_pt)
print("a * b (element-wise) (NumPy)\n", a_np * b_np)
print("a * b (element-wise) (PyTorch)\n", a_pt * b_pt)
print("⚠️  '*' 는 element-wise. 행렬 곱이 아님 — 행렬 곱은 (3) 절 참고.")
print("")

# ────────────── (2) 벡터 내적 (dot product) ──────────────
# 두 1D 벡터의 같은 위치 곱의 합. (a*b).sum() 과 동치.
print("[2] 벡터 내적 (dot product)")

a_np = np.array([1., 2., 3.])
b_np = np.array([4., 5., 6.])
a_pt = torch.tensor([1., 2., 3.])
b_pt = torch.tensor([4., 5., 6.])

print(f"NumPy   np.dot(a, b)        = {np.dot(a_np, b_np)}")
print(f"PyTorch torch.dot(a, b)     = {torch.dot(a_pt, b_pt).item()}")
print(f"수동 계산 (a*b).sum()       = {(a_pt * b_pt).sum().item()}")
print("")

# 함정 : NumPy 의 np.dot 은 입력이 다차원이면 행렬 곱처럼 동작.
#        반면 torch.dot 은 1D 벡터에만 허용 — 다차원이면 에러.
A = np.array([[1., 2.], [3., 4.]])
B = np.array([[5., 6.], [7., 8.]])
print("np.dot(A, B)  ← NumPy는 2D 입력에서 matmul 처럼 동작")
print(np.dot(A, B))
print("→ PyTorch 에서 동일한 결과를 원하면 torch.matmul / @ 사용 (다음 절)")
print("")

# ────────────── (3) 행렬 곱 (matmul) ──────────────
# 차원 규칙 : (M, K) × (K, N) → (M, N)
print("[3] 행렬 곱 (matmul)")

A_np = np.random.rand(3, 4).astype(np.float32)
B_np = np.random.rand(4, 2).astype(np.float32)
A_pt = torch.from_numpy(A_np)
B_pt = torch.from_numpy(B_np)

C_np = np.matmul(A_np, B_np)               # 또는 A_np @ B_np
C_pt = torch.matmul(A_pt, B_pt)            # 또는 A_pt @ B_pt

print(f"NumPy   shape : A{A_np.shape} @ B{B_np.shape} = C{C_np.shape}")
print(f"PyTorch shape : A{tuple(A_pt.shape)} @ B{tuple(B_pt.shape)} = C{tuple(C_pt.shape)}")
print(f"두 결과 동일 여부 : {np.allclose(C_np, C_pt.numpy())}")
print("")

# 배치 행렬 곱 : (B, M, K) × (B, K, N) → (B, M, N)
# 신경망에서 흔히 쓰이는 패턴 (배치 축이 가장 앞에 있음)
A_pt = torch.randn(8, 3, 4)                # 배치 8, 3x4 행렬
B_pt = torch.randn(8, 4, 2)                # 배치 8, 4x2 행렬
C_pt = torch.matmul(A_pt, B_pt)            # 배치 8, 3x2 행렬
print(f"배치 matmul : {tuple(A_pt.shape)} × {tuple(B_pt.shape)} = {tuple(C_pt.shape)}")
print("")

# ────────────── (4) Broadcasting ──────────────
# 규칙 : shape 의 끝에서부터 비교해 각 위치가 (같거나 / 한쪽이 1) 이면 자동 확장.
print("[4] Broadcasting")

# (3, 1) + (1, 4) → (3, 4) : 양쪽이 서로의 모자란 차원을 채움
a = torch.tensor([[1.], [2.], [3.]])       # shape (3, 1)
b = torch.tensor([[10., 20., 30., 40.]])   # shape (1, 4)
c = a + b
print(f"  shape  : (3, 1) + (1, 4) = {tuple(c.shape)}")
print(c)
print("")

# 흔한 예 : (N, D) 행렬에 (D,) 벡터를 더해 'bias' 추가
batch = torch.randn(5, 3)                  # (5, 3)
bias  = torch.tensor([1., 2., 3.])         # (3,)  → 자동으로 (1, 3) 으로 broadcast
print(f"  (5, 3) + (3,) = {tuple((batch + bias).shape)}   ← 신경망의 'y = x @ W + b' 패턴")
print("")

# ★ 함정 : (N,) 와 (N, 1) 의 차이 — 의도치 않은 (N, N) 발생
v_1d  = torch.tensor([1., 2., 3., 4.])     # shape (4,)
v_col = v_1d.view(-1, 1)                   # shape (4, 1)
v_row = v_1d.view(1, -1)                   # shape (1, 4)

# 의도 : '같은 위치끼리 더하기' 라고 생각했지만…
result_wrong = v_col + v_row               # (4, 1) + (1, 4) = (4, 4)  ← outer-sum!
print(f"  v_col + v_row : (4, 1) + (1, 4) = {tuple(result_wrong.shape)}  ← 의도치 않은 4x4")
print(result_wrong)
print("")

# 올바른 의도였다면 :
result_right_a = v_1d + v_1d                # 같은 (4,) 끼리 → (4,)
result_right_b = v_col + v_col              # 같은 (4, 1) 끼리 → (4, 1)
print(f"  v_1d + v_1d   : (4,) + (4,)     = {tuple(result_right_a.shape)}")
print(f"  v_col + v_col : (4,1) + (4,1)   = {tuple(result_right_b.shape)}")
print("→ 차원이 미묘하게 다른 두 텐서를 더하기 전에 항상 .shape 를 확인할 것.")
print("")

# ────────────── 비교 정리 ──────────────
# 작업                          NumPy                       PyTorch
# ───────────────────────────  ─────────────────────────  ───────────────────────
# Element-wise +,-,*,/,**       a + b 등 (모두 동일)       a + b 등
# 벡터 내적 (1D, 1D)             np.dot(a, b)               torch.dot(a, b) ← 1D 만 허용
# 행렬 곱 (M,K) × (K,N)         np.matmul(A, B)            torch.matmul(A, B)
#                                A @ B                       A @ B
# 배치 행렬 곱                   np.matmul (앞쪽 배치 축)    torch.matmul / torch.bmm
# 외적 (1D, 1D → 2D)             np.outer(a, b)             torch.outer(a, b)
# 트랜스포즈                     A.T  /  np.transpose(A)    A.T  /  torch.transpose(A, ...)
# 일반 텐서 곱 (Einstein 표기)   np.einsum('ij,jk->ik',A,B) torch.einsum('ij,jk->ik', A, B)
#
# 흔한 함정
# - '*' 는 element-wise. 행렬 곱은 '@' 또는 matmul.
# - np.dot 은 다차원이면 matmul 처럼 동작 (자동 변환). torch.dot 은 1D 강제.
# - (N,) 과 (N, 1) 은 broadcasting 결과가 다름 — 의도치 않은 (N, N) 발생 주의.
# - 신경망에서 'bias 더하기' 는 (N, D) + (D,) 형태의 broadcasting 패턴.
