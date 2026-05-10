# 08_conv_special.py
# Convolution 응용 — Linear vs Conv, Receptive Field, 1x1 Conv
#
# 학습 목표
#   "왜 영상에는 Conv 가 효율적인가" 와 "깊은 네트워크의 이점" 을 파라미터 수로 직접 확인.

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) Linear vs Conv — 파라미터 수 비교 ──────────────
# 1채널 128×128 → 3채널 64×64 변환을 두 방식으로.
print("[1] Linear vs Conv — 같은 입출력 변환")

inp_4d = torch.randn(4, 1, 128, 128)

# Linear : 모든 픽셀을 전부 연결
flat = inp_4d.reshape(4, -1)                                 # (4, 16384)
linear = nn.Linear(128*128, 64*64*3, bias=False)
out_linear = linear(flat).reshape(4, 3, 64, 64)
print(f"  Linear : 파라미터 수 = {get_num_params(linear):>12,d}")
print(f"           입력 {tuple(inp_4d.shape)} → 출력 {tuple(out_linear.shape)}")

# Conv : 같은 커널이 sliding (kernel=3, stride=2, padding=1 → 크기 절반)
conv = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1, bias=False)
out_conv = conv(inp_4d)
print(f"  Conv   : 파라미터 수 = {get_num_params(conv):>12,d}    ← 차이가 천 배 이상")
print(f"           입력 {tuple(inp_4d.shape)} → 출력 {tuple(out_conv.shape)}")
print("  → Conv 는 같은 커널을 입력 전체에 sliding 하므로 입력 크기가 커도 파라미터가 늘지 않음")
print("")

# ────────────── (2) Receptive Field — 깊이 vs 큰 커널 ──────────────
# 3x3 두 번 = 5x5 한 번 (RF 같음). 그런데 파라미터와 비선형성은 다르다.
print("[2] Receptive Field — 3x3 두 번 vs 5x5 한 번")

# 다이어그램 (1차원 단순화)
#   입력          : - - - - - - - -
#   3x3 conv 1회  :   ◯ ◯ ◯ ◯ ◯ ◯       (한 ◯ = 입력 3 픽셀)
#   3x3 conv 2회  :       ● ● ● ●         (한 ● = ◯ 3개 = 입력 5 픽셀)
#   결과적으로 ● 한 칸 = 입력 5칸 (= 5x5 conv 1회와 동일)

inp = torch.randn(1, 1, 256, 256)

model_a = nn.Sequential(                          # 3x3 두 번
    nn.Conv2d(1, 1, kernel_size=3, bias=False),
    nn.Conv2d(1, 1, kernel_size=3, bias=False),
)
out_a = model_a(inp)
print(f"  3x3 두 번 : 파라미터 {get_num_params(model_a)} (= 9+9), 출력 {tuple(out_a.shape)}")

model_b = nn.Conv2d(1, 1, kernel_size=5, bias=False)   # 5x5 한 번
out_b = model_b(inp)
print(f"  5x5 한 번 : 파라미터 {get_num_params(model_b)} (= 25),   출력 {tuple(out_b.shape)}")
print("  → 같은 RF 를 더 적은 파라미터 + 더 깊은 비선형성으로 표현 (현대 CNN 의 기본 원리)")
print("")

# ────────────── (3) 1x1 Convolution — 채널 변환기 ──────────────
# kernel_size=1 은 H, W 는 건드리지 않고 채널 수만 바꿈.
# 채널 간 선형 결합 + 비선형성 추가에 자주 사용 (ResNet, Inception 등의 핵심 빌딩 블록).
print("[3] 1x1 Convolution — 채널만 변환")

inp = torch.randn(1, 1, 256, 256)

# 채널 1 → 3
layer = nn.Conv2d(1, 3, kernel_size=1, bias=True)
out = layer(inp)
print(f"  Conv2d(1, 3, kernel=1) : 파라미터 {get_num_params(layer)} (= 3*1*1*1 + 3)")
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(out.shape)}    ← H, W 그대로, 채널만 1→3")

# 채널 3 → 1
layer = nn.Conv2d(3, 1, kernel_size=1, bias=True)
inp = torch.randn(1, 3, 256, 256)
out = layer(inp)
print(f"  Conv2d(3, 1, kernel=1) : 파라미터 {get_num_params(layer)} (= 1*3*1*1 + 1)")
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(out.shape)}    ← H, W 그대로, 채널만 3→1")

# ────────────── 비교 정리 ──────────────
# - Linear vs Conv : 영상처럼 큰 입력에는 Conv 가 압도적으로 효율적 (파라미터 수)
# - Receptive Field : 같은 RF 를 깊이로 만드는 것이 파라미터/비선형성 측면에서 유리
# - 1x1 Conv :
#     · H, W 는 그대로, 채널 수만 변환
#     · 채널 간 선형 결합으로 정보 압축/확장
#     · ResNet 의 bottleneck, Inception 모듈 등에서 핵심 부품
