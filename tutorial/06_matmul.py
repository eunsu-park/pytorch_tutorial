# 06_matmul.py
# 벡터 내적과 행렬 곱 — NumPy ↔ PyTorch
# 신경망에서 가장 자주 쓰이는 연산. Linear, attention, conv 의 본질이 모두 행렬 곱.

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) 벡터 내적 (dot product) ──────────────
# 두 1D 벡터의 같은 위치 곱의 합. (a*b).sum() 과 동치.
print("[1] 벡터 내적")

# NumPy
a_np = np.array([1., 2., 3.])
b_np = np.array([4., 5., 6.])
print(f"  NumPy   : np.dot(a, b)    = {np.dot(a_np, b_np)}")
print(f"            (a * b).sum()   = {(a_np * b_np).sum()}")

# PyTorch
a_pt = torch.tensor([1., 2., 3.])
b_pt = torch.tensor([4., 5., 6.])
print(f"  PyTorch : torch.dot(a, b) = {torch.dot(a_pt, b_pt).item()}")
print(f"            (a * b).sum()   = {(a_pt * b_pt).sum().item()}")
print("")

# ⚠️ 함정 : np.dot 은 다차원이면 matmul 처럼 동작하지만, torch.dot 은 1D 만 허용.
A_np = np.array([[1., 2.], [3., 4.]])
B_np = np.array([[5., 6.], [7., 8.]])
print("  np.dot(A, B)  ← NumPy 는 2D 입력에서 matmul 처럼 자동 동작")
print(np.dot(A_np, B_np))
try:
    torch.dot(torch.from_numpy(A_np), torch.from_numpy(B_np))
except RuntimeError as e:
    print("  torch.dot(A, B) ← PyTorch 는 1D 만 허용 → RuntimeError")
    print("                    다차원이면 torch.matmul / @ 사용")
print("")

# ────────────── (2) 행렬 곱 (matmul) ──────────────
# 차원 규칙 : (M, K) × (K, N) → (M, N)
print("[2] 행렬 곱")

# NumPy
A_np = np.random.randn(3, 4)
B_np = np.random.randn(4, 2)
C1_np = np.matmul(A_np, B_np)
C2_np = A_np @ B_np                          # @ 연산자 (PEP 465) — NumPy 도 지원
print(f"  NumPy   : A{A_np.shape} @ B{B_np.shape} = C{C1_np.shape}")
print(f"            np.matmul 과 @ 동일 : {np.allclose(C1_np, C2_np)}")

# PyTorch
A_pt = torch.from_numpy(A_np)
B_pt = torch.from_numpy(B_np)
C1_pt = torch.matmul(A_pt, B_pt)
C2_pt = A_pt @ B_pt
print(f"  PyTorch : A{tuple(A_pt.shape)} @ B{tuple(B_pt.shape)} = C{tuple(C1_pt.shape)}")
print(f"            torch.matmul 과 @ 동일 : {torch.allclose(C1_pt, C2_pt)}")
print(f"            NumPy 결과와 동일 : {np.allclose(C1_np, C1_pt.numpy())}")
print("")

# 차원 mismatch 시도 (양쪽 모두 에러)
try:
    bad = torch.randn(3, 4) @ torch.randn(5, 2)
except RuntimeError:
    print("  (3,4) @ (5,2) → RuntimeError (양쪽 모두). 내부 차원이 일치(K=K)해야 함.")
print("")

# ────────────── (3) 배치 행렬 곱 ──────────────
# (B, M, K) × (B, K, N) → (B, M, N).  배치 단위 학습에서 가장 흔한 패턴.
print("[3] 배치 행렬 곱")

# NumPy : np.matmul 이 앞쪽 차원을 배치로 처리
A_np = np.random.randn(8, 3, 4)
B_np = np.random.randn(8, 4, 2)
C_np = np.matmul(A_np, B_np)
print(f"  NumPy   : {A_np.shape} @ {B_np.shape} = {C_np.shape}")

# PyTorch : torch.matmul 또는 torch.bmm
A_pt = torch.from_numpy(A_np)
B_pt = torch.from_numpy(B_np)
C_pt_matmul = torch.matmul(A_pt, B_pt)
C_pt_bmm    = torch.bmm(A_pt, B_pt)             # bmm 은 (B, M, K) × (B, K, N) 정확히 그 모양만
print(f"  PyTorch : {tuple(A_pt.shape)} @ {tuple(B_pt.shape)} = {tuple(C_pt_matmul.shape)}")
print(f"            matmul 과 bmm 동일  : {torch.allclose(C_pt_matmul, C_pt_bmm)}")
print("")

# ────────────── (4) Linear 레이어의 본질 ──────────────
# nn.Linear(in, out) 의 forward 는 결국 x @ W.T + b 형태.
print("[4] Linear 의 본질 — x @ W.T + b")

x = torch.randn(5, 8)
W = torch.randn(4, 8)
b = torch.randn(4)
y = x @ W.T + b
print(f"  x{tuple(x.shape)} @ W.T{tuple(W.T.shape)} + b{tuple(b.shape)} = y{tuple(y.shape)}")
print("  ← nn.Linear(8, 4) 의 forward 와 정확히 동일한 계산")

# ────────────── 비교 정리 ──────────────
# 작업                       NumPy                          PyTorch
# ────────────────────────  ─────────────────────────────  ────────────────────────────
# 벡터 내적 (1D, 1D)         np.dot(a, b)                    torch.dot(a, b)  ← 1D 전용
# 행렬 곱 (M,K) × (K,N)     np.matmul(A, B) / A @ B         torch.matmul(A, B) / A @ B
# 배치 행렬 곱               np.matmul (앞 차원이 배치)       torch.matmul / torch.bmm
# 외적 (1D, 1D → 2D)         np.outer(a, b)                  torch.outer(a, b)
# Einstein 합                np.einsum('ij,jk->ik', A, B)    torch.einsum('ij,jk->ik', A, B)
#
# 흔한 함정
# - '*' 는 element-wise. 행렬 곱은 '@' 또는 matmul (양쪽 모두).
# - np.dot 은 다차원에서 matmul 처럼 동작 → torch.dot 으로 옮길 때 주의 (1D 전용).
# - 차원 mismatch (K != K) 는 RuntimeError. 항상 .shape 로 확인.
