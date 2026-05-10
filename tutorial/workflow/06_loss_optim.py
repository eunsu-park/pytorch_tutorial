# 06_loss_optim.py
# 모델 파라미터 — 손실 함수 / 옵티마이저 / 스케줄러
#
# 세 가지를 한 챕터에 모은 이유 : 학습 루프에서 항상 한 묶음으로 등장하기 때문.
#   loss = criterion(output, target)
#   optimizer.zero_grad(); loss.backward(); optimizer.step()
#   scheduler.step()                 # 매 epoch 끝
# 다음 챕터(07)에서 이 묶음으로 실제 학습 루프를 돌린다.

import torch
import torch.nn as nn
import torch.optim as optim


# ────────────── (1) 손실 함수 — criterion ──────────────
print("[1] 손실 함수")

# 회귀
mse  = nn.MSELoss()                                    # 평균 제곱 오차 — 가장 표준
mae  = nn.L1Loss()                                     # 평균 절대 오차 — 이상치에 강건

# 이진분류 (출력 1개)
bce        = nn.BCELoss()                              # 모델이 sigmoid 까지 적용한 확률을 받음
bce_logits = nn.BCEWithLogitsLoss()                    # 모델이 logit 을 받음 (수치 안정 — 권장)
bce_pos    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))   # 클래스 불균형 보정

# 다중분류 (출력 K개)
ce       = nn.CrossEntropyLoss()                       # logit 을 받음. 모델 마지막에 Softmax 두지 않음
ce_class = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 2.0])) # 클래스별 가중치

print(f"  MSELoss               : {mse}")
print(f"  L1Loss                : {mae}")
print(f"  BCEWithLogitsLoss     : {bce_logits}    ← 이진분류 권장")
print(f"  CrossEntropyLoss      : {ce}             ← 다중분류 권장")
print("")

# 사용자 정의 — 함수 형태
def relative_error_loss(pred, target):
    """상대 오차의 제곱 평균"""
    return ((pred - target) / (target.abs() + 1e-6)).pow(2).mean()


# 사용자 정의 — nn.Module 형태 (학습 가능 파라미터나 버퍼가 있을 때)
class WeightedMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)         # buffer : 학습 안 되지만 device 따라감
    def forward(self, pred, target):
        return ((pred - target) ** 2 * self.weight).mean()


print("  사용자 정의 손실 함수 — 함수 또는 nn.Module 둘 다 가능")
print("")

# ────────────── (2) 옵티마이저 — optimizer ──────────────
# 학습 가능한 파라미터를 받아 .grad 를 사용해 갱신.
print("[2] 옵티마이저")

# 학습할 모델 (예시용)
model = nn.Linear(10, 1)

# SGD — 가장 단순. 영상 분류 등에서 여전히 강력 (특히 momentum + scheduler 조합)
opt_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam — 파라미터별 적응형 lr. GAN/Transformer 등에서 표준
opt_adam = optim.Adam(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)

# AdamW — Adam + decoupled weight decay (Transformer/LLM 학습 표준)
opt_adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

print(f"  SGD     : {opt_sgd.defaults}")
print(f"  Adam    : lr={opt_adam.defaults['lr']}, betas={opt_adam.defaults['betas']}")
print(f"  AdamW   : lr={opt_adamw.defaults['lr']}, weight_decay={opt_adamw.defaults['weight_decay']}")
print("")

# 파라미터 그룹별 다른 학습률 — pretrained backbone + new head 조합에서 자주 사용
class TwoPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 5)
        self.head     = nn.Linear(5, 1)
    def forward(self, x): return self.head(self.backbone(x))


model_2 = TwoPart()
opt_groups = optim.Adam([
    {"params": model_2.backbone.parameters(), "lr": 1e-5},     # backbone 은 천천히
    {"params": model_2.head.parameters(),     "lr": 1e-3},     # head 는 빠르게
])
print(f"  파라미터 그룹별 lr : {[g['lr'] for g in opt_groups.param_groups]}")
print("")

# ────────────── (3) 학습률 스케줄러 — scheduler ──────────────
# 학습이 진행되면서 lr 을 줄이는 정책. 매 epoch 끝에서 scheduler.step() 호출.
print("[3] 스케줄러")

# StepLR — 일정 주기마다 감소 (가장 흔함)
sch_step = optim.lr_scheduler.StepLR(opt_adam, step_size=30, gamma=0.1)

# MultiStepLR — 정해진 epoch 에서만 감소
sch_multi = optim.lr_scheduler.MultiStepLR(opt_adam, milestones=[60, 120], gamma=0.1)

# ExponentialLR — 매 epoch 일정 비율 감소
sch_exp = optim.lr_scheduler.ExponentialLR(opt_adam, gamma=0.95)

# CosineAnnealingLR — 코사인 곡선 감소 (Transformer/Vision 표준)
sch_cos = optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=100, eta_min=1e-6)

# ReduceLROnPlateau — 검증 지표가 개선 안 되면 감소 (적응형)
sch_plat = optim.lr_scheduler.ReduceLROnPlateau(opt_adam, mode="min", factor=0.5, patience=5, min_lr=1e-7)
# 사용 시 : scheduler.step(val_loss)

print("  StepLR / MultiStepLR / ExponentialLR / CosineAnnealingLR / ReduceLROnPlateau")
print("  매 epoch 끝에 scheduler.step() 호출. ReduceLROnPlateau 만 .step(val_loss) 형태.")

# ────────────── 정리 ──────────────
# 선택 기준
#   손실 함수
#     회귀                   : MSELoss / L1Loss
#     이진분류               : BCEWithLogitsLoss   (모델 출력 = logit)
#     다중분류               : CrossEntropyLoss    (모델 출력 = logit, target = 클래스 인덱스)
#     영상 변환 (pix2pix 등) : L1Loss + 적대적 손실
#
#   옵티마이저
#     영상 분류              : SGD + Momentum + scheduler
#     생성 모델 (GAN)        : Adam (beta1=0.5)
#     Transformer/LLM        : AdamW + Cosine scheduler
#
#   스케줄러
#     단순한 학습             : StepLR (30, 60, 90 epoch 에서 1/10)
#     검증 지표 보고 결정      : ReduceLROnPlateau
#     Transformer/Vision      : CosineAnnealingLR (warm-up + 코사인 감소)
