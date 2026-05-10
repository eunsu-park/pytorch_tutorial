# 02_regression_nn.py
# 선형 회귀 — nn.Linear + nn.MSELoss 로 가장 추상화된 형태
#
# 01 과 비교
#   01 : w, b 두 tensor 를 직접 만들고 곱·합으로 모델을 표현
#   02 : nn.Linear(1, 1) 한 줄로 동일한 모델을 정의 (내부에 weight, bias 자동 등록)
# 손실 함수도 직접 구현 → nn.MSELoss() 로 교체.

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)

# ────────────── 데이터 생성 ──────────────
def generate_dataset(w_true=0.9, b_true=0.3, num=200):
    x = np.linspace(0, 100, num).reshape(-1, 1).astype(np.float32)
    noise = np.random.normal(0, 5, size=x.shape).astype(np.float32)
    y = w_true * x + b_true + noise
    return x, y.astype(np.float32)

x_np, y_np = generate_dataset()
x = torch.from_numpy(x_np)
y = torch.from_numpy(y_np)

# ────────────── 모델·손실·옵티마이저 ──────────────
model = nn.Linear(1, 1, bias=True)            # y_pred = w * x + b 와 동일한 계산
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0002)

print(model)
print(loss_fn)
print(optimizer)
print("")

# ────────────── 학습 — 표준 4단계 ──────────────
EPOCHS = 5000
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    y_pred = model(x)                          # forward
    loss = loss_fn(y_pred, y)                  # 손실 계산
    loss.backward()                            # 미분 자동 계산
    optimizer.step()                           # 파라미터 갱신

    if (epoch + 1) % 1000 == 0:
        w = model.weight.item()
        b = model.bias.item()
        print(f"  epoch {epoch+1:>5d} : w={w:.4f}, b={b:.4f}, loss={loss.item():.4f}")

# 최종 결과
w = model.weight.item()
b = model.bias.item()
print("")
print(f"최종 w={w:.4f}, b={b:.4f}    (정답 w=0.9, b=0.3)")

# ────────────── 비교 정리 ──────────────
# 단계 추상화 비교
#
#                         15 (수동 + autograd)            16 (nn.Linear + MSELoss)
# ──────────────────────  ─────────────────────────────  ──────────────────────────
# 모델 정의                w = torch.zeros(1, ...)         model = nn.Linear(1, 1)
#                          b = torch.zeros(1, ...)
# 순전파                   y_pred = w * x + b              y_pred = model(x)
# 손실 함수                ((y_pred - y) ** 2).mean()      loss_fn = nn.MSELoss();
#                                                          loss = loss_fn(y_pred, y)
# 옵티마이저 등록           SGD([w, b], lr=...)            SGD(model.parameters(), lr=...)
# 학습 루프 4단계           동일                            동일 (zero_grad/forward/backward/step)
#
# 핵심
#   - nn.Linear 같은 빌트인 레이어를 쓰면 가중치 등록·초기화·.parameters() 가 자동
#   - 다음 챕터(17~) 부터는 이 4단계 학습 루프를 그대로 두고 모델 부분만 점점 풍성하게 한다
