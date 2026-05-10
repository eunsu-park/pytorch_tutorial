# 13_loss_optimizer.py
# 손실 함수 + 옵티마이저 — 학습 루프에서 한 짝으로 동작하는 두 요소
#
# 학습 표준 4 단계 (02 챕터에서 본 그대로)
#     optimizer.zero_grad()
#     output = model(x)
#     loss   = loss_fn(output, y)
#     loss.backward()
#     optimizer.step()

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)


# ────────────── (1) 자주 쓰는 손실 함수 ──────────────
print("[1] 손실 함수")

# 회귀
pred   = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.5, 2.5, 2.5, 5.0])
print(f"  MSELoss      : {nn.MSELoss()(pred, target).item():.4f}    회귀 표준")
print(f"  L1Loss       : {nn.L1Loss()(pred, target).item():.4f}    이상치에 강건")

# 이진분류
logits_bin = torch.tensor([[2.0], [-1.0], [0.5], [-3.0]])
y_bin      = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
print(f"  BCEWithLogitsLoss(logits, y) : {nn.BCEWithLogitsLoss()(logits_bin, y_bin).item():.4f}    이진분류 (sigmoid 포함)")

# 다중분류
logits_mc = torch.tensor([[2.0, 0.5, -1.0],
                          [0.1, 1.5,  0.3],
                          [0.0, 0.0,  3.0]])
y_mc      = torch.tensor([0, 1, 2])         # 정답 클래스 인덱스 (one-hot 아님)
print(f"  CrossEntropyLoss(logits, y)  : {nn.CrossEntropyLoss()(logits_mc, y_mc).item():.4f}    다중분류 (softmax 포함)")
print("CrossEntropyLoss 는 logit 을 받음. 모델 마지막에 Softmax 를 두지 않는다 (내부에서 log_softmax 처리).")
print("")

# ────────────── (2) 옵티마이저 — SGD / Momentum / Adam ──────────────
# 같은 회귀 문제를 세 옵티마이저로 풀어 학습 속도 비교
print("[2] 옵티마이저 비교 — 같은 회귀 문제, 같은 모델 초기값")

def generate_dataset(num=200):
    x = np.linspace(0, 1, num).reshape(-1, 1).astype(np.float32)
    noise = np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
    y = (0.9 * x + 0.3 + noise).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

x, y = generate_dataset()


def train_one(name, num_steps=1000):
    """주어진 옵티마이저로 학습. 100 step 단위 손실 history 반환."""
    torch.manual_seed(0)                              # 모델 초기값 고정 → 공정한 비교
    model   = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    if name == "SGD":           opt = torch.optim.SGD(model.parameters(), lr=0.1)
    elif name == "SGD+Mom":     opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    elif name == "Adam":        opt = torch.optim.Adam(model.parameters(), lr=0.01)

    history = []
    for step in range(num_steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            history.append(loss.item())
    return history


print(f"  {'step':>5}  {'SGD(0.1)':>10}  {'SGD+Mom':>10}  {'Adam(.01)':>10}")
hs, hm, ha = train_one("SGD"), train_one("SGD+Mom"), train_one("Adam")
for i, (a, b, c) in enumerate(zip(hs, hm, ha)):
    print(f"  {(i+1)*100:>5d}  {a:>10.4f}  {b:>10.4f}  {c:>10.4f}")
print("")

# ────────────── 비교 정리 ──────────────
# 손실 함수 선택 기준
#   회귀 (연속값)             nn.MSELoss / nn.L1Loss            모델 출력 = 실수
#   이진분류 (확률 1개)       nn.BCEWithLogitsLoss              모델 출력 = logit 1개
#   다중분류 (K 중 하나)      nn.CrossEntropyLoss               모델 출력 = logit K개 (softmax 없이!)
#   영상→영상 변환            nn.L1Loss (+ 적대적 손실)         Pix2Pix 등
#
# 옵티마이저 선택 기준
#   SGD                  : 가장 단순. lr 민감. 일반화 좋을 때가 많음 (영상 분류 표준).
#   SGD + Momentum       : 진동 줄이고 수렴 가속.
#   Adam                 : 파라미터별 적응형 lr. 일반적으로 빠르게 수렴 (GAN 등 표준).
#
# 학습률 가이드 (lr)
#   SGD   : 1e-2 ~ 1e-1 정도
#   Adam  : 1e-4 ~ 1e-3 정도가 일반적
