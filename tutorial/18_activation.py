# 18_activation.py
# 활성 함수(Activation Function) — 곡선 시각화 + 모델 안 사용
#
# 활성 함수가 없으면 신경망은 결국 선형 변환의 합성이라 표현력이 제한된다.
# 비선형성을 더해주는 것이 활성 함수의 역할.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) 대표 활성 함수 ──────────────
# 모두 학습 가능한 파라미터가 없음 (nn.ReLU 등은 함수와 다름없음).
print("[1] 활성 함수의 파라미터 수")

relu    = nn.ReLU()        # y = max(0, x) — 가장 표준
sigmoid = nn.Sigmoid()     # y = 1 / (1 + exp(-x)) — [0, 1]
tanh    = nn.Tanh()        # y = (e^x - e^-x)/(e^x + e^-x) — [-1, 1]

for name, layer in [("ReLU", relu), ("Sigmoid", sigmoid), ("Tanh", tanh)]:
    print(f"  {name:<8s} 파라미터 수 = {get_num_params(layer)}")
print("")

# ────────────── (2) 곡선 모양 비교 ──────────────
# 1D 입력으로 그려야 곡선 모양이 명확히 보임.
print("[2] 곡선 시각화 (matplotlib 창이 뜸)")

x = torch.linspace(-5, 5, 200).view(-1, 1)
plt.plot(x.numpy(), relu(x).numpy(),    label="ReLU",    color="r")
plt.plot(x.numpy(), sigmoid(x).numpy(), label="Sigmoid", color="g")
plt.plot(x.numpy(), tanh(x).numpy(),    label="Tanh",    color="b")
plt.axhline(0, color="k", linewidth=0.5)
plt.axvline(0, color="k", linewidth=0.5)
plt.legend()
plt.grid(True)
plt.title("Activation Functions")
plt.show()

# ────────────── (3) 모델 안에서 사용 ──────────────
# 일반적인 패턴 : Linear → 활성 → Linear → 활성 ...
# 마지막 레이어의 활성 선택은 문제 종류에 따라 다름 (회귀=없음/Identity, 분류=Softmax 등).
print("[3] Sequential 안에서 사용")

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),                        # 은닉층 활성 — ReLU 가 표준
    nn.Linear(4, 3),
    nn.Sigmoid(),                     # 출력층 활성 — 이진분류 등에서
)
print(model)
print(f"  총 파라미터 수 : {get_num_params(model)}")

inp = torch.randn(128, 2)
out = model(inp)
print(f"  {tuple(inp.shape)} → {tuple(out.shape)},  값 범위 = [{out.min().item():.3f}, {out.max().item():.3f}]")
print("  ← 마지막 Sigmoid 때문에 출력이 (0, 1) 범위")

# ────────────── 비교 정리 ──────────────
# 자주 쓰이는 활성 함수
#   ReLU       y = max(0, x)                   은닉층 표준
#   LeakyReLU  y = x (x>0), αx (x<0)            ReLU 의 음수 영역 살림 (GAN 등)
#   Sigmoid    y = 1/(1+e^-x), [0, 1]            이진분류 출력 / 확률
#   Tanh       y = (e^x-e^-x)/(e^x+e^-x), [-1, 1]  정규화된 영상 출력 (Pix2Pix 등)
#   Softmax    각 dim 합이 1                     다중분류의 추론 단계에서 확률 변환 (학습 시에는 CrossEntropyLoss 가 내부에서 처리)
#
# 참고 : 활성 함수는 학습 가능한 파라미터가 0 이지만, 모델의 표현력에 큰 영향을 미침.
