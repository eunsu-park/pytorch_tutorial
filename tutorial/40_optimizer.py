# 40_optimizer.py
# 옵티마이저(Optimizer) 비교 — SGD / Momentum / Adam
#
# 학습 목표
#   같은 회귀 문제를 세 옵티마이저로 풀어 학습 속도와 수렴 양상을 비교.
#   17_regression_torch_1.py 의 후속 챕터로 봐도 좋다.

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)

# ────────────── 데이터 생성 ──────────────
# 데이터 스케일이 크면 옵티마이저별 lr 민감도가 커지므로 입력을 [0, 1] 로 정규화.
def generate_dataset(w_true=0.9, b_true=0.3, num=200):
    x = np.linspace(0, 1, num).reshape(-1, 1).astype(np.float32)
    noise = np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
    y = (w_true * x + b_true + noise).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

x, y = generate_dataset()

# ────────────── 학습 함수 ──────────────
def train_one(optimizer_name, num_steps=1000, lr=None):
    """
    한 옵티마이저로 num_steps 만큼 학습하고 100 step 단위 손실을 반환.
    lr 은 옵티마이저별로 권장값을 사용 (Adam 은 SGD 보다 보통 작은 lr).
    """
    torch.manual_seed(0)                       # 모델 초기값 고정 → 옵티마이저 비교가 공정해짐
    model = nn.Linear(1, 1, bias=True)
    loss_fn = nn.MSELoss()

    if optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr or 0.1)
    elif optimizer_name == "SGD+Momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr or 0.1, momentum=0.9)
    elif optimizer_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr or 0.01)
    else:
        raise ValueError(optimizer_name)

    history = []
    for step in range(num_steps):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            history.append(loss.item())
    return history


# ────────────── 비교 실행 ──────────────
print("[Loss every 100 steps]  (옵티마이저별 권장 lr 사용)")
print(f"{'step':>5}  {'SGD(0.1)':>10}  {'SGD+Mom':>10}  {'Adam(.01)':>10}")

h_sgd  = train_one("SGD")
h_mom  = train_one("SGD+Momentum")
h_adam = train_one("Adam")

for i, (a, b, c) in enumerate(zip(h_sgd, h_mom, h_adam)):
    print(f"{(i+1)*100:>5d}  {a:>10.4f}  {b:>10.4f}  {c:>10.4f}")

print("")

# ────────────── 비교 정리 ──────────────
# - SGD            : 가장 단순. lr 에 민감하고 수렴이 느림. 최종 일반화는 좋을 때가 많음.
# - SGD + Momentum : 이전 step 의 방향을 일정 비율(0.9 등) 유지 → 진동을 줄이고 수렴 가속.
# - Adam           : 파라미터별 적응형 lr. 일반적으로 빠르게 수렴. 본 예시처럼 단순 회귀에서는
#                    같은 lr 로 비교하면 SGD 보다 한참 빠르게 손실이 감소함.
#
# 실전 팁
# - Adam 의 기본 lr 은 1e-3 ~ 2e-4 정도가 일반적. SGD 는 더 큰 lr 이 필요할 수 있음.
# - 영상 분류는 SGD + Momentum + LR scheduler 조합이 여전히 강력 (예: ResNet 학습).
# - GAN/생성 모델은 Adam(beta1=0.5) 가 표준 (generation/train.py 참고).
