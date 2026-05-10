# 03_linear.py
# Linear 레이어 — 단일 사용 + Sequential 로 쌓기
#
# 핵심 공식
#   nn.Linear(in_features, out_features) 의 학습 파라미터 수
#     = in_features * out_features  (weight)  +  out_features  (bias)

import torch
import torch.nn as nn


def get_num_params(model):
    """모델/레이어의 학습 가능 파라미터 수 (이후 챕터에서도 반복 사용)"""
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) 단일 Linear 레이어 ──────────────
print("[1] 단일 Linear")

linear = nn.Linear(2, 4)              # 입력 2 → 출력 4
print(linear)
# 파라미터 수 : 2*4 + 4 = 12
print(f"  파라미터 수 : {get_num_params(linear)}    (= 2*4 weight + 4 bias)")

# 입력 (배치 N, in_features) → 출력 (N, out_features)
inp = torch.randn(128, 2)
out = linear(inp)
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(out.shape)}")

# 다른 배치 크기에도 자동 대응 (배치 차원은 가변)
inp = torch.randn(256, 2)
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(linear(inp).shape)}")
print("")

# ────────────── (2) Sequential 로 여러 Linear 쌓기 ──────────────
# nn.Sequential : 여러 레이어를 순서대로 적용. 작은 모델에 적합.
print("[2] Sequential 로 쌓기")

model = nn.Sequential(
    nn.Linear(2, 4),                  # 12 파라미터
    nn.Linear(4, 3),                  # 4*3 + 3 = 15 파라미터
)
print(model)
print(f"  총 파라미터 수 : {get_num_params(model)}    (= 12 + 15)")

inp = torch.randn(128, 2)
print(f"  {tuple(inp.shape)} → {tuple(model(inp).shape)}")
print("")

# ────────────── (3) Sequential 의 동등 표기 ──────────────
# 리스트로 쌓아두고 *unpack 하는 패턴도 자주 쓰임 (조건부 추가 등에 편리).
print("[3] 동등 표기")

layers = []
layers += [nn.Linear(8, 16)]
layers += [nn.Linear(16, 4)]
model2 = nn.Sequential(*layers)
print(model2)
print(f"  파라미터 수 : {get_num_params(model2)}")
print("")

# ────────────── 비교 정리 ──────────────
# - nn.Linear(in, out)
#     · 내부적으로 weight (out, in), bias (out,) 등록
#     · forward 는 사실상 x @ weight.T + bias (tensor/06 챕터 참고)
#     · 입력 마지막 차원이 in_features 와 같아야 함, 배치 차원은 가변
# - 파라미터 수 공식 :  in * out + out   (bias=False 면 -out)
# - Sequential
#     · 내부 모듈을 등록 순서대로 호출
#     · 짧고 단순한 모델에 적합. 분기/skip-connection 은 nn.Module 상속 (11 챕터)
