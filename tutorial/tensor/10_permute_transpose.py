# 10_permute_transpose.py
# 차원 순서 변경 — NumPy ↔ PyTorch
#
# 주의 : 같은 함수 이름 'transpose' 가 NumPy 와 PyTorch 에서 의미가 다르다.
#     NumPy   의 np.transpose : 모든 차원 재배치 (= PyTorch 의 permute)
#     PyTorch 의 torch.transpose : 두 차원만 교환

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) 모든 차원 재배치 ──────────────
# 영상 처리에서 가장 흔한 변환 (H, W, C) → (C, H, W).
print("[1] 모든 차원 재배치")

# NumPy : np.transpose(x, axes) — '새 위치 → 원래 위치' 매핑
x_np = np.random.rand(32, 32, 3)            # (H, W, C)
y_np = np.transpose(x_np, (2, 0, 1))         # (C, H, W)
print(f"  NumPy   : 원본 {x_np.shape}")
print(f"            np.transpose(x, (2,0,1)) → {y_np.shape}    ← (C, H, W) 변환")

# PyTorch : torch.permute(t, dims) — 동일한 동작
x_pt = torch.randn(32, 32, 3)
y_pt = torch.permute(x_pt, (2, 0, 1))
y_pt2 = x_pt.permute(2, 0, 1)               # 메서드 형태도 가능
print(f"  PyTorch : 원본 {tuple(x_pt.shape)}")
print(f"            torch.permute(x, (2,0,1)) → {tuple(y_pt.shape)}    ← 같은 결과")
print(f"            x.permute(2, 0, 1)         → {tuple(y_pt2.shape)}")
print("")

# ────────────── (2) 두 차원만 교환 ──────────────
print("[2] 두 차원 교환")

# NumPy : np.swapaxes 또는 np.transpose 의 인자 줄이기
x_np = np.random.randn(2, 3, 4, 5)
y_np = np.swapaxes(x_np, 1, 3)               # 1번과 3번 교환
print(f"  NumPy   : np.swapaxes(x, 1, 3) : {x_np.shape} → {y_np.shape}")
print(f"            x.T (2D)              : {np.random.randn(3, 4).shape} → {np.random.randn(3, 4).T.shape}")

# PyTorch : torch.transpose — 두 차원만 교환 (NumPy 의 transpose 와 의미 다름!)
x_pt = torch.randn(2, 3, 4, 5)
y_pt = torch.transpose(x_pt, 1, 3)
print(f"  PyTorch : torch.transpose(x, 1, 3) : {tuple(x_pt.shape)} → {tuple(y_pt.shape)}")
x_pt_2d = torch.randn(3, 4)
print(f"            x.t() (2D 단축)            : {tuple(x_pt_2d.shape)} → {tuple(x_pt_2d.t().shape)}")
print(f"            x.T (NumPy 와 같은 표기)   : {tuple(x_pt_2d.shape)} → {tuple(x_pt_2d.T.shape)}")
print("")

# ────────────── (3) 같은 이름, 다른 의미 ──────────────
# 'transpose' 라는 이름이 두 라이브러리에서 다른 의미를 가진다는 점이 핵심.
print("[3] transpose 의미 차이")

# NumPy : np.transpose(x, (...)) — 모든 차원 재배치
arr = np.random.randn(2, 3, 4)
print(f"  np.transpose(arr, (2, 0, 1))    → {np.transpose(arr, (2, 0, 1)).shape}    ← 모든 차원 재배치")

# PyTorch : torch.transpose(t, a, b) — 두 차원만 교환
ten = torch.randn(2, 3, 4)
print(f"  torch.transpose(ten, 0, 2)      → {tuple(torch.transpose(ten, 0, 2).shape)}    ← 두 차원만 교환")
print(f"  torch.permute(ten, (2, 0, 1))   → {tuple(torch.permute(ten, (2, 0, 1)).shape)}    ← NumPy 의 transpose 와 같은 의미")
print("")

# ────────────── (4) 영상 데이터 표준 변환 ──────────────
# 데이터 로딩 후 학습에 넣기 전 거의 항상 한 번 등장.
print("[4] 영상 데이터 표준 변환 (H, W, C) → (C, H, W)")

# OpenCV / Matplotlib / scikit-image 가 주는 형식 → PyTorch 가 기대하는 형식
img_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
img_chw_np = np.transpose(img_np, (2, 0, 1))
img_pt = torch.from_numpy(img_np)
img_chw_pt = img_pt.permute(2, 0, 1)
print(f"  NumPy   : (H, W, C) {img_np.shape} → (C, H, W) {img_chw_np.shape}")
print(f"  PyTorch : (H, W, C) {tuple(img_pt.shape)} → (C, H, W) {tuple(img_chw_pt.shape)}")

# ────────────── 비교 정리 ──────────────
# 작업                           NumPy                              PyTorch
# ────────────────────────────  ────────────────────────────────  ─────────────────────────────
# 모든 차원 재배치                np.transpose(x, axes)              torch.permute(t, dims)
#                                                                     t.permute(*dims)
# 두 차원 교환                    np.swapaxes(x, a, b)               torch.transpose(t, a, b)
# 2D 전치 (단축)                  x.T                                t.T  /  t.t()
#
# 주의 : 'transpose' 의 의미가 다름
#     NumPy   의 np.transpose      = '모든 차원 재배치'  (PyTorch 의 permute)
#     PyTorch 의 torch.transpose   = '두 차원 교환'     (NumPy 의 swapaxes)
#
# 영상 데이터 관습
#   OpenCV/Matplotlib/scikit-image : (H, W, C)
#   PyTorch                        : (C, H, W)
#   → 학습 전 np.transpose(img, (2,0,1)) 또는 img.permute(2, 0, 1) 호출
#
# 결과의 메모리는 비연속이 될 수 있음 → view 안 되는 경우 reshape 또는 .contiguous() 필요
