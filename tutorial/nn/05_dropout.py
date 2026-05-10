# 05_dropout.py
# Dropout — 학습 시 일부 뉴런을 무작위로 0 으로
#
# 핵심
#   - 학습(train) 시에만 적용 : 일부 뉴런을 확률 p 로 0 으로
#   - 평가(eval) 시에는 적용되지 않음 (전체 뉴런 사용)
#   - model.train() / model.eval() 로 모드 전환
#   - 학습 가능한 파라미터는 0 개

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) Dropout 단일 레이어 동작 ──────────────
print("[1] Dropout 단일 레이어")

dropout = nn.Dropout(p=0.5)           # 확률 p 로 0 으로
print(dropout)
print(f"  파라미터 수 : {get_num_params(dropout)}    ← 학습 가능 파라미터 없음")
print("")

# 입력의 절반 정도가 0 으로 바뀜. 0 이 안 된 값은 1/(1-p) 배로 스케일됨 (학습/평가 일관성을 위해)
torch.manual_seed(0)
inp = torch.ones(2, 6)
print(f"  inp        :\n{inp}")
print(f"  dropout(inp):\n{dropout(inp)}")    # 일부가 0, 나머지는 2 (=1/(1-0.5))
print("  → 0 이 아닌 값이 2 인 이유 : 1/(1-p) 로 스케일")
print("")

# ────────────── (2) Sequential 안에서 — train()/eval() 차이 ──────────────
# Dropout 은 'activation 후, 다음 Linear 전' 위치에 두는 것이 표준.
print("[2] train()/eval() 모드 차이")

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Dropout(p=0.5),                # 학습 시에만 동작
    nn.Linear(8, 4),
)
print(model)
print(f"  파라미터 수 : {get_num_params(model)}    (Linear 만 카운트됨)")
print("")

inp = torch.ones(1, 4)

# 학습 모드 : Dropout 적용
model.train()
torch.manual_seed(0)
out_train_1 = model(inp)
torch.manual_seed(1)
out_train_2 = model(inp)
print(f"  [train] 출력 1 : {out_train_1.detach().numpy().flatten()}")
print(f"  [train] 출력 2 : {out_train_2.detach().numpy().flatten()}")
print("  → seed 가 다르면 결과가 달라짐 (Dropout 은 무작위)")

# 평가 모드 : Dropout 미적용
model.eval()
out_eval_1 = model(inp)
out_eval_2 = model(inp)
print(f"  [eval]  출력 1 : {out_eval_1.detach().numpy().flatten()}")
print(f"  [eval]  출력 2 : {out_eval_2.detach().numpy().flatten()}")
print("  → 항상 같은 결과 (Dropout 이 비활성)")

# ────────────── 비교 정리 ──────────────
# - 목적 : 과적합 방지 — 학습 시 무작위로 일부 뉴런을 끔으로써 특정 뉴런에 과의존하지 않게
# - 위치 : 보통 활성함수 뒤, 다음 Linear/Conv 앞
# - 모드 :
#     model.train() → Dropout 적용 (학습 시)
#     model.eval()  → Dropout 미적용 (평가/추론 시)
# - 학습 가능 파라미터 0
# - 비슷한 변형 :
#     nn.Dropout2d   — 채널 단위 dropout (CNN 에서 가끔 사용)
#     nn.AlphaDropout — SELU 와 함께 쓰는 변형
#
# 주의사항
#   - inplace=True 는 입력을 직접 수정하므로 디버깅이 어려움 — 기본값(False) 권장
#   - 평가 루프 직전에 model.eval() 호출을 잊으면 결과가 매번 달라짐
