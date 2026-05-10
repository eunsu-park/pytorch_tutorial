# 10_permute_transpose.py
# 차원 순서 변경 — permute / transpose
#
# 가장 흔한 사용 :
#   영상 데이터 형식 변환 (H, W, C) ↔ (C, H, W)
#     NumPy/OpenCV/Matplotlib 관습 : (H, W, C)
#     PyTorch 관습                 : (C, H, W)

import torch

torch.manual_seed(0)

# ────────────── (1) permute — 모든 차원 재배치 ──────────────
# permute(*dims) 의 dims 는 '새 위치 → 원래 위치' 매핑.
print("[1] permute")

x = torch.randn(32, 32, 3)           # (H, W, C) 영상
print(f"  원본            : {tuple(x.shape)}    ← (H, W, C)")

# (H, W, C) → (C, H, W) : 새 0축 = 원래 2축, 새 1축 = 원래 0축, 새 2축 = 원래 1축
y = x.permute(2, 0, 1)
print(f"  permute(2,0,1)  : {tuple(y.shape)}    ← (C, H, W) 로 변환")
print("")

# ────────────── (2) transpose — 두 차원만 교환 ──────────────
# torch.transpose(t, a, b) : 차원 a 와 b 만 서로 바꿈.
print("[2] transpose")

x = torch.randn(3, 4)
print(f"  원본       : {tuple(x.shape)}")
print(f"  x.t()      : {tuple(x.t().shape)}    ← 2D 전용 단축 (NumPy 의 .T 와 같음)")
print(f"  transpose  : {tuple(torch.transpose(x, 0, 1).shape)}    ← 차원 0 과 1 교환")
print("")

# 4D tensor 의 두 차원만 교환
x4 = torch.randn(2, 3, 4, 5)
y4 = torch.transpose(x4, 1, 3)       # 1번과 3번만 교환
print(f"  (2,3,4,5) transpose(1, 3) → {tuple(y4.shape)}    (= (2,5,4,3))")
print("")

# ────────────── (3) ⚠️  PyTorch 와 NumPy 의 transpose 의미 차이 ──────────────
# NumPy 의 np.transpose 는 '모든 차원 재배치' (PyTorch 의 permute 와 같음).
# PyTorch 의 torch.transpose 는 '두 차원 교환만' 가능.
print("[3] 라이브러리별 차이")
print("  NumPy    : np.transpose(x, axes)  → 모든 차원 재배치 (PyTorch 의 permute)")
print("  PyTorch  : torch.transpose(t, a, b)  → 두 차원만 교환")
print("  PyTorch  : torch.permute(t, dims) / t.permute(*dims) → 모든 차원 재배치")
print("")

# ────────────── (4) 영상 처리 표준 변환 ──────────────
# 데이터 로딩 후 학습에 넣기 전 거의 항상 한 번 등장.
print("[4] 영상 데이터 표준 변환")

img_hwc = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8)   # OpenCV 가 주는 형식
img_chw = img_hwc.permute(2, 0, 1)                                   # PyTorch 가 기대하는 형식
print(f"  OpenCV (H, W, C) : {tuple(img_hwc.shape)}")
print(f"  → PyTorch (C, H, W) : {tuple(img_chw.shape)}    ← 학습 직전에 자주 호출")

# ────────────── 비교 정리 ──────────────
# - 모든 차원 재배치
#     NumPy   : np.transpose(x, (...))         ※ 모든 차원 가능
#     PyTorch : torch.permute(t, (...)) / t.permute(*dims)
# - 두 차원 교환
#     NumPy   : np.swapaxes(x, a, b)
#     PyTorch : torch.transpose(t, a, b)        ※ 의미가 NumPy 의 transpose 와 다름!
# - 영상 데이터 관습
#     OpenCV/Matplotlib  : (H, W, C)
#     PyTorch            : (C, H, W)            → 학습 전 permute(2, 0, 1)
# - permute / transpose 결과는 메모리 비연속 → view 안 되는 경우 reshape 또는 .contiguous()
