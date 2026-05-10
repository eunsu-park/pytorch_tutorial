# 39_loss.py
# 손실 함수(Loss Function) — 자주 쓰는 함수들의 비교
#
# 학습 목표
#   회귀 vs 이진분류 vs 다중분류 각각에 어떤 손실 함수가 적합한지 익히고,
#   nn.BCELoss / nn.BCEWithLogitsLoss / nn.CrossEntropyLoss 의 입력 차이를 본다.

import torch
import torch.nn as nn

torch.manual_seed(0)

# ────────────── (1) 회귀 — MSE / L1 / SmoothL1 ──────────────
print("[1] 회귀 손실")

pred   = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.5, 2.5, 2.5, 5.0])

mse  = nn.MSELoss()
l1   = nn.L1Loss()
sl1  = nn.SmoothL1Loss()
print(f"MSELoss      : {mse(pred, target).item():.4f}    (큰 오차에 민감, 학습이 빠름)")
print(f"L1Loss       : {l1(pred, target).item():.4f}    (이상치에 강건)")
print(f"SmoothL1Loss : {sl1(pred, target).item():.4f}    (작은 오차는 L2, 큰 오차는 L1 — 둘의 절충)")
print("")

# ────────────── (2) 이진분류 — BCE / BCEWithLogits ──────────────
print("[2] 이진분류 손실")

logits = torch.tensor([[2.0], [-1.0], [0.5], [-3.0]])   # 모델 출력(logit)
y      = torch.tensor([[1.0], [0.0], [1.0], [0.0]])     # 정답 (0 또는 1)

probs  = torch.sigmoid(logits)                          # sigmoid 를 거치면 [0, 1] 확률

bce        = nn.BCELoss()
bce_logits = nn.BCEWithLogitsLoss()                     # 내부에서 sigmoid + BCE 를 한 번에 (수치 안정)
print(f"BCELoss(sigmoid(logits), y)    : {bce(probs, y).item():.4f}")
print(f"BCEWithLogitsLoss(logits, y)   : {bce_logits(logits, y).item():.4f}")
print("→ 두 값은 같지만, BCEWithLogitsLoss 가 수치적으로 더 안정적이므로 권장")
print("")

# ────────────── (3) 다중분류 — CrossEntropy / NLL ──────────────
print("[3] 다중분류 손실")

logits = torch.tensor([[2.0, 0.5, -1.0],
                       [0.1, 1.5,  0.3],
                       [0.0, 0.0,  3.0]])
target = torch.tensor([0, 1, 2])                        # 정답 클래스 인덱스 (one-hot 아님)

ce  = nn.CrossEntropyLoss()                             # 내부 : log_softmax + NLLLoss
nll = nn.NLLLoss()
print(f"CrossEntropyLoss(logits, target)         : {ce(logits, target).item():.4f}")
print(f"NLLLoss(log_softmax(logits), target)     : {nll(torch.log_softmax(logits, 1), target).item():.4f}")
print("→ 두 값 동일. CrossEntropyLoss 는 logit 을 받고, NLLLoss 는 log_softmax 를 받음.")
print("⚠️  모델 출력에 nn.Softmax 를 직접 두면 CrossEntropyLoss 와 이중 적용 — 자세한 건 44 챕터")
print("")

# ────────────── 비교 정리 ──────────────
# 작업 종류                    추천 손실                     모델 출력
# ─────────────────────────  ─────────────────────────  ───────────────
# 회귀 (연속값)              nn.MSELoss / nn.L1Loss        실수
# 이진분류 (확률 1개)        nn.BCEWithLogitsLoss          logit 1개
# 다중분류 (K개 중 하나)     nn.CrossEntropyLoss           logit K개 (softmax 없이)
# 다중분류 (모델이 logsoft.) nn.NLLLoss                    log_softmax 출력
# 영상→영상 변환             nn.L1Loss (+ 적대적 손실)     영상 (예: pix2pix)
