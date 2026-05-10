# 29_softmax_pitfall.py
# CrossEntropyLoss 와 Softmax — '이중 softmax' 함정
#
# 핵심 한 줄
#   nn.CrossEntropyLoss = log_softmax(logits) → NLLLoss(target)
# 즉 CrossEntropyLoss 는 logit(생 점수)을 입력으로 기대한다.
# 모델 출력에 Softmax 를 거치면 softmax 가 두 번 적용되어 학습이 비정상이 됨.

import torch
import torch.nn as nn

torch.manual_seed(0)

# 가짜 분류 문제 : batch=4, num_classes=3
logits = torch.randn(4, 3, requires_grad=True)
target = torch.tensor([0, 1, 2, 1])           # 정답 클래스 인덱스 (one-hot 아님)

# ────────────── (1) CrossEntropyLoss 의 내부 동작 분해 ──────────────
print("[1] CrossEntropyLoss 분해")

ce = nn.CrossEntropyLoss()
loss_ce = ce(logits, target)
print(f"  CrossEntropyLoss(logits, target)            = {loss_ce.item():.6f}")

# 같은 결과를 직접 만들기 : log_softmax + NLLLoss
log_probs = torch.log_softmax(logits, dim=1)
nll = nn.NLLLoss()
loss_manual = nll(log_probs, target)
print(f"  NLLLoss(log_softmax(logits), target)        = {loss_manual.item():.6f}")

print(f"  두 결과 동일 여부 : {torch.allclose(loss_ce, loss_manual)}")
print("")

# ────────────── (2) 함정 — 모델 출력에 Softmax 를 붙인 경우 ──────────────
print("[2] 함정 — Softmax + CrossEntropyLoss 동시 사용")

probs = torch.softmax(logits, dim=1)          # 모델이 마지막에 softmax 를 거친 출력 (확률)
loss_wrong = ce(probs, target)                # 여기서 CrossEntropyLoss 는 softmax 를 또 적용함
print(f"  잘못된 사용  : CrossEntropyLoss(softmax(logits), target) = {loss_wrong.item():.6f}")
print(f"  올바른 사용  : CrossEntropyLoss(logits, target)          = {loss_ce.item():.6f}")
print("  → 두 값이 다르고, 잘못된 사용 쪽이 손실이 비정상적으로 작아져 학습 신호가 약해짐")
print("")

# ────────────── (3) 올바른 패턴 ──────────────
print("[3] 올바른 모델 출력 형식")

# 패턴 A : 모델은 logit 만 출력 + CrossEntropyLoss        ← 권장
class ModelA(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(8, 3)
    def forward(self, x):
        return self.fc(x)                     # logit 그대로 (Softmax 없음)

# 패턴 B : 모델이 log_softmax 를 직접 출력 + NLLLoss
class ModelB(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(8, 3)
    def forward(self, x):
        return torch.log_softmax(self.fc(x), dim=1)

x = torch.randn(4, 8)
y = torch.tensor([0, 1, 2, 1])
print(f"  패턴 A : ModelA + CrossEntropyLoss   loss = {nn.CrossEntropyLoss()(ModelA()(x), y).item():.4f}")
print(f"  패턴 B : ModelB + NLLLoss            loss = {nn.NLLLoss()(ModelB()(x), y).item():.4f}")
print("")

# ────────────── 비교 정리 ──────────────
# - nn.CrossEntropyLoss : logit 을 받음 (내부에 log_softmax 포함). 모델 출력에 softmax 두지 말 것.
# - nn.NLLLoss          : log_softmax 를 받음. 모델이 직접 log_softmax 출력해야 함.
# - 추론 시 확률이 필요하면 학습 후 별도로 softmax 적용 :
#       with torch.no_grad():
#           probs = torch.softmax(model(x), dim=1)
#
# 실전 체크리스트
# - [ ] 분류 모델의 마지막에 nn.Softmax 를 두지 않았는가?
# - [ ] CrossEntropyLoss 를 쓰면 모델 출력은 logit 인가?
# - [ ] target 의 형식은 클래스 인덱스 (LongTensor) 인가? (one-hot 아님)
#
# 참고 : classification/networks.py 에 본 함정이 의도적으로 보존되어 있음.
#        해당 파일의 ⚠️ 주석을 확인하고, 어떤 부분이 잘못된 패턴인지 직접 찾아볼 것.
