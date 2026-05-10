# 15_regression_manual.py
# 선형 회귀 — 수동 vs autograd 비교
#
# 학습 흐름
#   왼쪽(NumPy)  : 손실의 미분식을 손으로 유도해 매 step 직접 적용
#   오른쪽(PyTorch + autograd) : loss.backward() 한 줄로 미분이 자동 계산됨
# 같은 데이터, 같은 학습률, 같은 epoch 수에서 두 방식이 같은 결과를 내는지 확인.

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── 데이터 생성 (양쪽 동일) ──────────────
def generate_dataset(w_true=0.9, b_true=0.3, num=200):
    x = np.linspace(0, 100, num).reshape(-1, 1).astype(np.float32)
    noise = np.random.normal(0, 5, size=x.shape).astype(np.float32)
    y = w_true * x + b_true + noise
    return x, y.astype(np.float32)

x_np, y_np = generate_dataset()
x_pt = torch.from_numpy(x_np)
y_pt = torch.from_numpy(y_np)

LR = 0.0002
EPOCHS = 5000

# ────────────── (1) NumPy — 수동 gradient descent ──────────────
print("[1] NumPy 수동 (직접 미분식 작성)")

# MSE 의 미분식 :  dL/dw = (2/N) * sum((y_pred - y) * x),  dL/db = (2/N) * sum(y_pred - y)
w, b = 0.0, 0.0
for epoch in range(EPOCHS):
    y_pred = w * x_np + b
    grad_w = 2 * ((y_pred - y_np) * x_np).mean()
    grad_b = 2 *  (y_pred - y_np).mean()
    w -= LR * grad_w
    b -= LR * grad_b
loss_np = ((w * x_np + b - y_np) ** 2).mean()
print(f"  최종 w={w:.4f}, b={b:.4f}, loss={loss_np:.4f}")
print("")

# ────────────── (2) PyTorch — autograd 가 미분 ──────────────
print("[2] PyTorch + autograd (backward 가 미분 담당)")

w = torch.zeros(1, requires_grad=True)       # learnable 파라미터
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=LR)

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    y_pred = w * x_pt + b
    loss = ((y_pred - y_pt) ** 2).mean()      # MSE
    loss.backward()                           # 미분 자동 계산
    optimizer.step()                          # w, b 업데이트

print(f"  최종 w={w.item():.4f}, b={b.item():.4f}, loss={loss.item():.4f}")
print("")

# ────────────── 비교 정리 ──────────────
# 두 방식의 결과는 동일해야 함 (같은 lr, 같은 epoch, 같은 데이터, 같은 random seed).
# 차이점은 'w, b 의 grad 를 누가 계산하느냐' 뿐.
#
# 수동 (NumPy)                             autograd (PyTorch)
# ───────────────────────────────────────  ──────────────────────────────────
# 미분식을 사람이 손으로 쓴다              loss.backward() 가 자동 계산
# w, b 는 일반 변수                        requires_grad=True 로 추적 대상 표시
# w -= lr * grad_w                          optimizer.step() 이 갱신
# 한 줄 한 줄 명시적                       모델이 복잡해져도 같은 4단계로 동작 :
#                                              optimizer.zero_grad()
#                                              loss = ...
#                                              loss.backward()
#                                              optimizer.step()
#
# 실전에서는 모델이 복잡하면 미분식을 손으로 쓰는 게 불가능 → autograd 가 필수.
# 다음 챕터(16) : 모델 정의도 nn.Linear 로 더 추상화한다.
