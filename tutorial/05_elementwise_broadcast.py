# 05_elementwise_broadcast.py
# Element-wise 연산과 Broadcasting

import torch

torch.manual_seed(0)

# ────────────── (1) Element-wise 연산 ──────────────
# 같은 shape 끼리 +, -, *, /, ** 등은 위치별로 계산.
print("[1] Element-wise")

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
b = torch.tensor([[10., 10., 10.], [20., 20., 20.]])

print("a + b\n", a + b)
print("a * b (element-wise)\n", a * b)
print("⚠️  '*' 는 element-wise. 행렬 곱이 아님 — 행렬 곱은 06 챕터의 '@' 또는 matmul.")
print("")

# ────────────── (2) Broadcasting 의 규칙 ──────────────
# shape 의 끝에서부터 비교해 각 위치가 (같거나 / 한쪽이 1) 이면 자동 확장.
print("[2] Broadcasting 규칙")

# (3, 1) + (1, 4) → (3, 4) : 양쪽이 서로의 모자란 차원을 채움
a = torch.tensor([[1.], [2.], [3.]])           # (3, 1)
b = torch.tensor([[10., 20., 30., 40.]])       # (1, 4)
c = a + b                                      # (3, 4)
print(f"  (3, 1) + (1, 4) = {tuple(c.shape)}")
print(c)
print("")

# 신경망 표준 패턴 : (N, D) + (D,) → 'bias 더하기'
batch = torch.randn(5, 3)
bias  = torch.tensor([1., 2., 3.])             # (3,) 가 (1, 3) 으로 자동 확장
print(f"  (N, D) + (D,) : (5, 3) + (3,) = {tuple((batch + bias).shape)}    ← y = x @ W + b 패턴")
print("")

# ────────────── (3) ★ 함정 — (N,) vs (N, 1) ──────────────
# 같아 보이지만 broadcasting 결과는 전혀 다름.
print("[3] 함정 : (N,) 과 (N, 1)")

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

# ────────────── (4) 명시적 broadcast 확인 ──────────────
# torch.broadcast_shapes 로 결과 shape 미리 계산 가능
shape = torch.broadcast_shapes((3, 1, 5), (1, 4, 5))
print(f"  broadcast_shapes((3,1,5), (1,4,5)) = {shape}")

# ────────────── 비교 정리 ──────────────
# - element-wise : 양쪽 라이브러리 동일. '*' 는 절대 행렬 곱이 아님.
# - broadcasting 규칙 : 끝 차원부터 (같음 or 한쪽이 1) 이면 자동 확장
# - 흔한 함정 :
#     · (N,) vs (N, 1) — outer-sum 으로 (N, N) 만들 수 있음
#     · (1, N) vs (N,) — 결과 같지만 의도 명확화를 위해 view/unsqueeze 권장
# - 의도 명확화 : reshape / unsqueeze 로 차원을 명시적으로 맞추는 습관
