# 05_elementwise_broadcast.py
# Element-wise 연산과 Broadcasting — NumPy ↔ PyTorch
# 두 라이브러리 모두 거의 동일한 문법과 규칙을 따른다.

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) Element-wise — 같은 shape 끼리 ──────────────
print("[1] Element-wise (양쪽 동일 동작)")

# NumPy
a_np = np.array([[1., 2., 3.], [4., 5., 6.]])
b_np = np.array([[10., 10., 10.], [20., 20., 20.]])
print("  NumPy   a + b :\n", a_np + b_np)
print("  NumPy   a * b :\n", a_np * b_np)

# PyTorch
a_pt = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
b_pt = torch.tensor([[10., 10., 10.], [20., 20., 20.]])
print("  PyTorch a + b :\n", a_pt + b_pt)
print("  PyTorch a * b :\n", a_pt * b_pt)
print("'*' 는 element-wise. 행렬 곱이 아님 — 행렬 곱은 06 챕터의 '@' 또는 matmul.")
print("")

# ────────────── (2) Broadcasting 규칙 ──────────────
# shape 의 끝에서부터 비교해 각 위치가 (같거나 / 한쪽이 1) 이면 자동 확장.
# NumPy 와 PyTorch 의 규칙은 동일.
print("[2] Broadcasting 규칙")

# NumPy : (3, 1) + (1, 4) → (3, 4)
a_np = np.array([[1.], [2.], [3.]])           # (3, 1)
b_np = np.array([[10., 20., 30., 40.]])       # (1, 4)
print(f"  NumPy   (3,1) + (1,4) = {(a_np + b_np).shape}")
print(a_np + b_np)

# PyTorch : 동일 결과
a_pt = torch.tensor([[1.], [2.], [3.]])
b_pt = torch.tensor([[10., 20., 30., 40.]])
print(f"  PyTorch (3,1) + (1,4) = {tuple((a_pt + b_pt).shape)}")
print(a_pt + b_pt)
print("")

# 신경망 표준 패턴 : (N, D) + (D,) — 'bias 더하기'
batch_np = np.random.randn(5, 3)
bias_np  = np.array([1., 2., 3.])
batch_pt = torch.randn(5, 3)
bias_pt  = torch.tensor([1., 2., 3.])
print(f"  NumPy   (N,D) + (D,) : {(batch_np + bias_np).shape}")
print(f"  PyTorch (N,D) + (D,) : {tuple((batch_pt + bias_pt).shape)}    ← y = x @ W + b 패턴")
print("")

# ────────────── (3) (N,) vs (N, 1) — 차원이 미묘하게 다른 두 텐서 ──────────────
# 양쪽 라이브러리 모두 broadcasting 결과가 같은 방식으로 동작한다.
print("[3] (N,) 과 (N, 1)")

# PyTorch 로 시연 (NumPy 도 결과 동일)
v_1d  = torch.tensor([1., 2., 3., 4.])         # shape (4,)
v_col = v_1d.view(-1, 1)                       # shape (4, 1)
v_row = v_1d.view(1, -1)                       # shape (1, 4)

# 의도 : '같은 위치끼리 더하기' 인 줄 알았는데…
result_wrong = v_col + v_row                   # (4, 1) + (1, 4) = (4, 4)  ← outer-sum!
print(f"  v_col + v_row  : (4, 1) + (1, 4) = {tuple(result_wrong.shape)}  ← 의도치 않은 4×4")
print(result_wrong)
print("")

# 올바른 의도였다면 :
print(f"  v_1d + v_1d    : (4,)  + (4,)  = {tuple((v_1d + v_1d).shape)}")
print(f"  v_col + v_col  : (4,1) + (4,1) = {tuple((v_col + v_col).shape)}")
print("→ 차원이 미묘하게 다른 두 텐서를 더하기 전에 항상 .shape 를 확인할 것.")
print("")

# NumPy 에서도 똑같이 동작
v_np_1d  = np.array([1., 2., 3., 4.])
v_np_col = v_np_1d.reshape(-1, 1)
v_np_row = v_np_1d.reshape(1, -1)
print(f"  NumPy  : (4,1) + (1,4) = {(v_np_col + v_np_row).shape}    ← 동일한 결과")
print("")

# ────────────── (4) 명시적 broadcast 확인 ──────────────
# 결과 shape 를 미리 계산하고 싶을 때
print("[4] broadcast shape 미리 확인")
print(f"  NumPy   : np.broadcast_shapes((3,1,5), (1,4,5))    = {np.broadcast_shapes((3,1,5), (1,4,5))}")
print(f"  PyTorch : torch.broadcast_shapes((3,1,5), (1,4,5)) = {torch.broadcast_shapes((3,1,5), (1,4,5))}")

# ────────────── 비교 정리 ──────────────
# - element-wise (+, -, *, /, **) : NumPy 와 PyTorch 가 완전히 동일 동작
# - '*' 는 절대 행렬 곱이 아님 (양쪽 모두) — 행렬 곱은 06 챕터
# - broadcasting 규칙 동일 : 끝 차원부터 (같음 or 한쪽이 1) 이면 자동 확장
# - (N,) vs (N, 1) 차이도 양쪽 모두 동일하게 동작
# - 의도 명확화 : reshape / unsqueeze / expand_dims 로 차원을 맞추는 습관
