# 09_unsqueeze_squeeze.py
# 차원 추가 / 제거 — NumPy ↔ PyTorch
#
# 자주 쓰이는 곳
#   · 단일 영상 → 배치 차원 추가 : (C, H, W) → (1, C, H, W)
#   · 배치 결과의 채널 차원 제거 : (N, 1, H, W) → (N, H, W)
#   · broadcasting 정렬 : (N,) → (N, 1) 로 만들어 (N, 1) + (1, M) = (N, M)

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) 차원 추가 ──────────────
print("[1] 차원 추가")

# NumPy
x_np = np.random.rand(32, 32)              # (32, 32)
y0_np = np.expand_dims(x_np, axis=0)        # (1, 32, 32)
y2_np = np.expand_dims(x_np, axis=2)        # (32, 32, 1)
print(f"  NumPy   : 원본 {x_np.shape}")
print(f"            np.expand_dims(x, axis=0) → {y0_np.shape}    ← 배치 차원 추가")
print(f"            np.expand_dims(x, axis=2) → {y2_np.shape}    ← 채널 차원을 끝에 추가")

# PyTorch
x_pt = torch.randn(32, 32)
y0_pt = torch.unsqueeze(x_pt, 0)
y2_pt = x_pt.unsqueeze(2)
y_neg = x_pt.unsqueeze(-1)                  # 음수 인덱스 = 끝에서부터
print(f"  PyTorch : 원본 {tuple(x_pt.shape)}")
print(f"            torch.unsqueeze(x, 0) → {tuple(y0_pt.shape)}    ← 같은 동작")
print(f"            x.unsqueeze(2)        → {tuple(y2_pt.shape)}    ← 메서드 형태")
print(f"            x.unsqueeze(-1)       → {tuple(y_neg.shape)}    ← 음수 인덱스 (NumPy 도 동일 지원)")
print("")

# 단일 영상 → 배치 차원 추가 (모델에 한 장만 넣을 때)
img_np = np.random.rand(3, 28, 28)
img_pt = torch.randn(3, 28, 28)
print(f"  NumPy   : 단일 영상 → 배치 : {img_np.shape} → {np.expand_dims(img_np, 0).shape}")
print(f"  PyTorch : 단일 영상 → 배치 : {tuple(img_pt.shape)} → {tuple(img_pt.unsqueeze(0).shape)}")
print("")

# ────────────── (2) 차원 제거 ──────────────
# 크기가 1 인 차원을 제거. 크기가 1 이 아닌 차원은 무시.
print("[2] 차원 제거")

# NumPy
y_np = np.random.rand(1, 3, 1, 32, 32)
print(f"  NumPy   : 원본 {y_np.shape}")
print(f"            np.squeeze(y)       → {np.squeeze(y_np).shape}     ← 크기 1인 모든 차원 제거")
print(f"            np.squeeze(y, 0)    → {np.squeeze(y_np, 0).shape}   ← 0번만 (크기 1)")
print(f"            np.squeeze(y, 2)    → {np.squeeze(y_np, 2).shape}     ← 2번만 (크기 1)")

# PyTorch
y_pt = torch.randn(1, 3, 1, 32, 32)
print(f"  PyTorch : 원본 {tuple(y_pt.shape)}")
print(f"            y.squeeze()          → {tuple(y_pt.squeeze().shape)}     ← 크기 1인 모든 차원 제거")
print(f"            y.squeeze(0)         → {tuple(y_pt.squeeze(0).shape)}   ← 0번만")
print(f"            y.squeeze(1)         → {tuple(y_pt.squeeze(1).shape)}  ← 1번 (크기 3) — 무시 (에러 아님)")
print("")

# 흔한 패턴 : 단일 채널 영상의 채널 차원 제거 후 시각화
out = torch.randn(8, 1, 28, 28)
single = out[0].squeeze(0)
print(f"  PyTorch : 배치 첫 영상 squeeze : {tuple(out[0].shape)} → {tuple(single.shape)}    ← matplotlib 시각화 가능 형태")
print("            (NumPy 도 동일 : np.squeeze(out[0], axis=0))")

# ────────────── 비교 정리 ──────────────
# 차원 추가
#     NumPy   : np.expand_dims(x, axis=k)
#     PyTorch : torch.unsqueeze(t, k)  /  t.unsqueeze(k)
# 차원 제거
#     NumPy   : np.squeeze(x, axis=k)        (axis 생략 시 크기 1인 차원 모두 제거)
#     PyTorch : torch.squeeze(t, k)  /  t.squeeze(k)   (k 생략 시 동일)
# 인자 이름 차이
#     NumPy 는 axis=, PyTorch 는 위치 인자 (또는 dim=)
# 공통
#     · 음수 인덱스 사용 가능 (-1 = 마지막)
#     · squeeze 는 크기가 1 이 아닌 차원에는 영향 없음 (안전)
#     · 단일 영상의 배치 차원 추가/제거가 가장 흔한 사용 사례
