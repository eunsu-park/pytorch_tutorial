# 07_conv_size.py
# Convolution 출력 크기 — padding, stride, 그리고 공식
#
# ★ 출력 크기 공식 (정사각형 가정)
#       H_out = floor( (H_in + 2 * padding - kernel_size) / stride ) + 1
#   ※ dilation=1 (기본값) 가정. 일반식은 PyTorch 공식 문서 참고.

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def expected_size(H_in, K, P, S):
    """공식으로 계산한 출력 크기"""
    return (H_in + 2 * P - K) // S + 1


# ────────────── (1) padding 의 효과 ──────────────
# padding 은 입력 영상의 가장자리를 0 으로 채워 넣음. 출력 크기를 보존하는 데 사용.
print("[1] padding")

inp = torch.randn(128, 1, 16, 16)

# padding=0 (기본) : 출력이 줄어듦
conv = nn.Conv2d(1, 3, kernel_size=3, padding=0)
out = conv(inp)
print(f"  padding=0 : 출력 {tuple(out.shape)}   "
      f"(공식: (16 + 0 - 3)/1 + 1 = {expected_size(16, 3, 0, 1)})")

# padding=1 : 출력 크기를 입력과 같게 유지 (kernel=3, padding=1 의 표준 조합)
conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
out = conv(inp)
print(f"  padding=1 : 출력 {tuple(out.shape)}  "
      f"(공식: (16 + 2 - 3)/1 + 1 = {expected_size(16, 3, 1, 1)})    ← 표준 조합")

# 비대칭 padding
conv = nn.Conv2d(1, 3, kernel_size=3, padding=(2, 1))
out = conv(inp)
print(f"  padding=(2,1) : 출력 {tuple(out.shape)}  ← H 와 W 에 다른 padding")
print("")

# ────────────── (2) stride 의 효과 ──────────────
# stride 는 커널이 한 번에 움직이는 칸 수. 출력 크기가 약 1/stride 로 줄어듦.
print("[2] stride")

inp = torch.randn(128, 1, 16, 16)

conv = nn.Conv2d(1, 3, kernel_size=3, stride=1)
out = conv(inp)
print(f"  stride=1 : 출력 {tuple(out.shape)}   "
      f"(공식: (16 + 0 - 3)/1 + 1 = {expected_size(16, 3, 0, 1)})")

conv = nn.Conv2d(1, 3, kernel_size=3, stride=2)
out = conv(inp)
print(f"  stride=2 : 출력 {tuple(out.shape)}     "
      f"(공식: (16 + 0 - 3)/2 + 1 = {expected_size(16, 3, 0, 2)})    ← 다운샘플링")
print("")

# ────────────── (3) 조합 — 다운샘플링 표준 ──────────────
# kernel=3, stride=2, padding=1 은 영상 크기를 정확히 절반으로 줄이는 표준 조합.
# Pix2Pix 의 U-Net Generator 나 ResNet stage 진입부에서 자주 사용.
print("[3] kernel=3, stride=2, padding=1 — H, W 를 절반으로")

inp = torch.randn(128, 1, 32, 32)
conv = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
out = conv(inp)
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(out.shape)}  "
      f"(공식: (32 + 2 - 3)/2 + 1 = {expected_size(32, 3, 1, 2)})")
print(f"  파라미터 수 : {get_num_params(conv)}")
print("")

# ────────────── (4) 공식 직접 사용 ──────────────
# 모델을 설계하기 전 출력 크기를 손으로 계산하는 습관이 중요하다.
print("[4] 공식 응용 — H_in=224, kernel=7, stride=2, padding=3")
print(f"  H_out = (224 + 2*3 - 7) / 2 + 1 = {expected_size(224, 7, 3, 2)}    ← ResNet 의 첫 conv")

# ────────────── 비교 정리 ──────────────
# 핵심 공식
#   H_out = floor( (H_in + 2*padding - kernel_size) / stride ) + 1
#
# 자주 쓰는 조합
#   kernel=3, padding=1                  → 크기 보존        (특징 추출 표준)
#   kernel=3, padding=1, stride=2        → 크기 1/2          (다운샘플링)
#   kernel=1                              → 크기 그대로, 채널만 변환 (다음 08 챕터)
#   kernel=4, padding=1, stride=2        → 크기 1/2          (Pix2Pix Discriminator 표준)
#
# 파라미터 수는 padding/stride 와 무관 — kernel_size, in_channels, out_channels 만 영향.
