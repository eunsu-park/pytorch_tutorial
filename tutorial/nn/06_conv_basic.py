# 06_conv_basic.py
# Convolution 기본 — kernel_size, in_channels, out_channels
#
# 핵심 파라미터 공식
#   nn.Conv2d(C_in, C_out, kernel_size=K) 의 학습 파라미터 수
#     = C_out * C_in * K * K  (weight)  +  C_out  (bias)
#   ※ bias=False 면 + C_out 부분 빠짐

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) kernel_size 변화 ──────────────
print("[1] kernel_size — 입력은 (B=128, C=1, H=16, W=16) 고정")

inp = torch.randn(128, 1, 16, 16)

for K in [3, 2, 1]:
    conv = nn.Conv2d(1, 1, kernel_size=K)
    out = conv(inp)
    # 파라미터 수 : 1 * 1 * K * K + 1
    print(f"  kernel={K}  → 파라미터 {get_num_params(conv):>3d}  (= 1*1*{K}*{K} + 1),  출력 {tuple(out.shape)}")

# 비대칭 커널도 가능
conv = nn.Conv2d(1, 1, kernel_size=(4, 2))
out = conv(inp)
print(f"  kernel=(4,2) → 파라미터 {get_num_params(conv)}  (= 1*1*4*2 + 1),  출력 {tuple(out.shape)}")
print("")

# ────────────── (2) in_channels, out_channels 변화 ──────────────
# Conv 는 입력의 모든 채널을 한꺼번에 보고, 그것을 out_channels 개의 새 채널로 변환.
# 출력 채널 수가 늘어도 H, W 는 그대로(이 절에서는 padding=0, stride=1).
print("[2] 채널 변화 (kernel=3 고정)")

# (1) 1 → 3 채널로 확장
conv = nn.Conv2d(1, 3, kernel_size=3)
inp = torch.randn(128, 1, 16, 16)
out = conv(inp)
# 파라미터 : 3 * 1 * 3 * 3 + 3 = 30
print(f"  Conv2d(1, 3) : 파라미터 {get_num_params(conv)}  (= 3*1*3*3 + 3)")
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(out.shape)}    ← 채널 1 → 3")

# (2) 3 → 1 채널로 축소
conv = nn.Conv2d(3, 1, kernel_size=3)
inp = torch.randn(128, 3, 16, 16)
out = conv(inp)
# 파라미터 : 1 * 3 * 3 * 3 + 1 = 28
print(f"  Conv2d(3, 1) : 파라미터 {get_num_params(conv)}  (= 1*3*3*3 + 1)")
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(out.shape)}    ← 채널 3 → 1")

# (3) 3 → 2 채널
conv = nn.Conv2d(3, 2, kernel_size=3)
inp = torch.randn(128, 3, 16, 16)
out = conv(inp)
# 파라미터 : 2 * 3 * 3 * 3 + 2 = 56
print(f"  Conv2d(3, 2) : 파라미터 {get_num_params(conv)}  (= 2*3*3*3 + 2)")
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(out.shape)}    ← 채널 3 → 2")
print("")

# ────────────── (3) 입력 영상 크기와 파라미터 수 ──────────────
# 입력의 H, W 가 바뀌어도 Conv 의 파라미터 수는 변하지 않음 (커널 크기만 영향).
# 단 출력의 H, W 는 입력에 따라 달라짐. 이는 07 챕터(padding/stride) 에서 자세히.
print("[3] 입력 크기 ↔ 파라미터 수 무관")

conv = nn.Conv2d(1, 1, kernel_size=3)
print(f"  conv 파라미터 수 : {get_num_params(conv)}")
for size in [16, 32, 64, 128]:
    inp = torch.randn(2, 1, size, size)
    print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(conv(inp).shape)}    ← 파라미터는 그대로")

# ────────────── 비교 정리 ──────────────
# - nn.Conv2d(C_in, C_out, kernel_size=K) 파라미터 수
#       = C_out * C_in * K * K + C_out
#   ※ bias=False 면 + C_out 부분 빠짐
# - 채널은 모든 입력 채널을 합산해 새 출력 채널을 만든다 (depthwise 가 아님)
# - 입력 H, W 가 달라져도 파라미터 수는 그대로 — 출력 크기만 달라짐
# - Linear 와 비교 : Conv 는 같은 커널이 입력 전 영역을 sliding 하므로
#   파라미터 수가 입력 H, W 에 비례하지 않음 → 영상에 효율적 (다음 챕터에서 자세히)
