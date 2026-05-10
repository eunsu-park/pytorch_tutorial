# 06_matmul.py
# 벡터 내적과 행렬 곱
#
# 신경망에서 가장 자주 쓰이는 연산. Linear 레이어, attention, conv 의 본질이 모두 행렬 곱.

import torch

torch.manual_seed(0)

# ────────────── (1) 벡터 내적 (dot product) ──────────────
# 두 1D 벡터의 같은 위치 곱의 합. (a*b).sum() 과 동치.
print("[1] 벡터 내적")

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
print(f"torch.dot(a, b)   = {torch.dot(a, b).item()}")
print(f"(a * b).sum()     = {(a * b).sum().item()}      ← 직접 계산해도 동일")
print("⚠️  torch.dot 은 1D 벡터에만 허용. 다차원이면 에러 — matmul 을 쓸 것.")
print("")

# ────────────── (2) 행렬 곱 (matmul) ──────────────
# 차원 규칙 : (M, K) × (K, N) → (M, N)
print("[2] 행렬 곱")

A = torch.randn(3, 4)
B = torch.randn(4, 2)
C1 = torch.matmul(A, B)
C2 = A @ B                       # @ 연산자 — 가장 권장되는 표기
print(f"A{tuple(A.shape)} @ B{tuple(B.shape)} = C{tuple(C1.shape)}")
print(f"matmul 과 @ 동일 여부 : {torch.allclose(C1, C2)}")
print("")

# 차원 mismatch 시도
try:
    bad = torch.randn(3, 4) @ torch.randn(5, 2)
except RuntimeError as e:
    print(f"  (3,4) @ (5,2) 시도 → RuntimeError")
    print(f"  내부 차원이 일치(K=K)해야 함")
print("")

# ────────────── (3) 배치 행렬 곱 ──────────────
# (B, M, K) × (B, K, N) → (B, M, N)
# 배치 단위 학습에서 가장 흔한 패턴.
print("[3] 배치 행렬 곱")

A = torch.randn(8, 3, 4)         # 배치 8, 3×4 행렬
B = torch.randn(8, 4, 2)         # 배치 8, 4×2 행렬
C = A @ B                        # 배치 8, 3×2 행렬
print(f"  {tuple(A.shape)} @ {tuple(B.shape)} = {tuple(C.shape)}")

# torch.bmm 은 정확히 (B, M, K) × (B, K, N) 만 받음. matmul 은 broadcasting 도 허용.
C_bmm = torch.bmm(A, B)
print(f"  torch.bmm 결과와 동일 : {torch.allclose(C, C_bmm)}")
print("")

# ────────────── (4) Linear 레이어의 본질 ──────────────
# nn.Linear(in, out) 의 forward 는 결국 x @ W.T + b 형태.
print("[4] Linear 의 본질 — x @ W.T + b")

x = torch.randn(5, 8)            # 배치 5, 입력 8
W = torch.randn(4, 8)            # weight (out, in) = (4, 8)
b = torch.randn(4)               # bias (4,)
y = x @ W.T + b                  # (5, 8) @ (8, 4) + (4,) → (5, 4)
print(f"  x{tuple(x.shape)} @ W.T{tuple(W.T.shape)} + b{tuple(b.shape)} = y{tuple(y.shape)}")
print("  ← nn.Linear(8, 4) 의 forward 와 정확히 동일한 계산")
print("")

# ────────────── 비교 정리 ──────────────
# 작업                          PyTorch
# ───────────────────────────  ───────────────────────
# 벡터 내적 (1D, 1D)             torch.dot(a, b)  ← 1D 만 허용
# 행렬 곱 (M,K) × (K,N)         torch.matmul(A, B)  /  A @ B  (권장)
# 배치 행렬 곱 (B,M,K)×(B,K,N)  torch.matmul / torch.bmm
# 외적 (1D, 1D → 2D)             torch.outer(a, b)
# Einstein 합 (모든 텐서 곱)     torch.einsum('ij,jk->ik', A, B)
#
# 흔한 함정
# - '*' 는 element-wise. 행렬 곱은 '@' 또는 matmul.
# - 차원 mismatch (K != K) 는 RuntimeError. 항상 .shape 로 확인.
# - Linear 레이어는 W 가 (out, in) 이므로 x @ W.T 형태로 계산됨.
