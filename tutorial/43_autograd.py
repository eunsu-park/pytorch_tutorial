# 43_autograd.py
# Autograd — PyTorch 가 자동으로 미분을 계산하는 방식
#
# 학습 목표
#   17_regression_torch_1.py 에서 `requires_grad=True`, `loss.backward()`,
#   `optimizer.step()` 을 사용했지만 내부에서 어떤 일이 일어나는지 본 챕터에서 정리한다.

import torch

# ────────────── (1) requires_grad 와 .grad ──────────────
print("[1] 단순한 1변수 미분")

x = torch.tensor(3.0, requires_grad=True)   # 이 tensor 에 대해 기울기를 추적
y = x ** 2 + 2 * x + 1                       # y = x^2 + 2x + 1

print(f"x = {x.item()}, y = {y.item()}")
y.backward()                                  # dy/dx 계산을 트리거
print(f"dy/dx (수동: 2x + 2 = 8)  → x.grad = {x.grad.item()}")
print("")

# ────────────── (2) 다변수 — 부분 미분 ──────────────
print("[2] 다변수 함수")

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
xv = torch.tensor(3.0)
yv = w * xv + b                               # y = w*x + b
yv.backward()
print(f"dy/dw (수동: x = 3)   → w.grad = {w.grad.item()}")
print(f"dy/db (수동: 1)        → b.grad = {b.grad.item()}")
print("")

# ────────────── (3) grad 누적 — zero_grad 의 이유 ──────────────
print("[3] grad 는 누적된다 (그래서 매 step zero_grad 호출)")

x = torch.tensor(1.0, requires_grad=True)
for step in range(3):
    y = x * 2
    y.backward()
    print(f"  step {step}: x.grad = {x.grad.item()}  (매 backward 마다 누적)")
x.grad.zero_()                                # 수동 초기화
print(f"  zero_() 후 : x.grad = {x.grad.item()}")
# 학습 루프에서 optimizer.zero_grad() 가 같은 일을 함
print("")

# ────────────── (4) 그래프 분리 — detach() ──────────────
print("[4] detach — 그래프 분리")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y.detach()                                 # z 는 grad 추적이 끊김
print(f"y.requires_grad = {y.requires_grad}, z.requires_grad = {z.requires_grad}")
# numpy 변환 시 detach 가 필요한 이유 : autograd 그래프가 있으면 numpy 변환이 막힘
print("")

# ────────────── (5) no_grad 컨텍스트 — 평가 시 메모리/속도 절약 ──────────────
print("[5] torch.no_grad — 추론 시 그래프 생성 안 함")

x = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y = x ** 2
print(f"no_grad 안의 y.requires_grad = {y.requires_grad}  (False 가 정상)")
# 추론(test.py) 에서 `with torch.no_grad():` 로 감싸는 이유

# ────────────── 비교 정리 ──────────────
# - requires_grad=True : 이 tensor 가 관여한 모든 연산을 추적
# - .backward()        : 결과 tensor 부터 거꾸로 미분 (그래프의 leaf 변수에 .grad 채움)
# - .grad              : 누적된 기울기 — 매 step optimizer.zero_grad() 로 초기화 필요
# - .detach()          : 그래프에서 tensor 를 떼어낸 사본 (값은 같음)
# - torch.no_grad()    : 컨텍스트 안에서는 그래프 생성 자체를 안 함 (추론 시 권장)
#
# 학습 루프의 표준 패턴 (17_regression_torch_1.py 와 비교)
#     optimizer.zero_grad()    # 이전 step 의 grad 를 0 으로
#     output = model(x)
#     loss   = loss_fn(output, y)
#     loss.backward()          # autograd → 모델 파라미터들의 .grad 채움
#     optimizer.step()         # .grad 를 사용해 파라미터 업데이트
