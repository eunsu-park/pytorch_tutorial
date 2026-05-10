# 11_cat_stack.py
# 텐서 합치기 — cat (기존 차원 따라) / stack (새 차원 만들며) — NumPy ↔ PyTorch
#
# 핵심 차이
#   cat   / concatenate : 이미 있는 차원의 길이가 늘어남
#   stack               : 새 차원이 하나 생김

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) 기존 차원 따라 합치기 ──────────────
# NumPy : np.concatenate, PyTorch : torch.cat. 인자 이름이 axis ↔ dim 으로 다름.
print("[1] 기존 차원 따라 — concatenate / cat")

# NumPy
x_np = np.random.rand(1, 32, 32)
y_np = np.random.rand(1, 32, 32)
z0_np = np.concatenate([x_np, y_np], axis=0)
z1_np = np.concatenate([x_np, y_np], axis=1)
print(f"  NumPy   : np.concatenate([x, y], axis=0) : {x_np.shape} + {y_np.shape} → {z0_np.shape}")
print(f"            np.concatenate([x, y], axis=1)                            → {z1_np.shape}")

# PyTorch
x_pt = torch.from_numpy(x_np)
y_pt = torch.from_numpy(y_np)
z0_pt = torch.cat([x_pt, y_pt], dim=0)
z1_pt = torch.cat([x_pt, y_pt], dim=1)
print(f"  PyTorch : torch.cat([x, y], dim=0) : {tuple(x_pt.shape)} + {tuple(y_pt.shape)} → {tuple(z0_pt.shape)}")
print(f"            torch.cat([x, y], dim=1)                            → {tuple(z1_pt.shape)}")
print("")

# 실전 예 : 채널 방향 결합 — Conditional GAN 의 입력 + 정답 쌍
inp = torch.randn(8, 1, 256, 256)
tar = torch.randn(8, 1, 256, 256)
pair = torch.cat([inp, tar], dim=1)
print(f"  pix2pix 입력 쌍 : torch.cat([inp, tar], dim=1) → {tuple(pair.shape)}")
print("")

# ────────────── (2) 새 차원 만들며 합치기 ──────────────
# 합칠 텐서들은 모두 같은 shape 이어야 함.
print("[2] 새 차원 만들며 — stack")

# NumPy
a_np = np.array([1., 2., 3.])
b_np = np.array([4., 5., 6.])
c_np = np.array([7., 8., 9.])
s0_np = np.stack([a_np, b_np, c_np], axis=0)
s1_np = np.stack([a_np, b_np, c_np], axis=1)
print(f"  NumPy   : np.stack([a, b, c], axis=0) → {s0_np.shape}\n{s0_np}")
print(f"            np.stack([a, b, c], axis=1) → {s1_np.shape}\n{s1_np}")

# PyTorch
a_pt = torch.from_numpy(a_np)
b_pt = torch.from_numpy(b_np)
c_pt = torch.from_numpy(c_np)
s0_pt = torch.stack([a_pt, b_pt, c_pt], dim=0)
print(f"  PyTorch : torch.stack([a, b, c], dim=0) → {tuple(s0_pt.shape)}    ← NumPy 와 같은 결과")
print(f"            결과 동일 여부 : {np.allclose(s0_np, s0_pt.numpy())}")
print("")

# ────────────── (3) cat vs stack — 직접 비교 ──────────────
print("[3] cat 과 stack 의 차이")

a_np = np.zeros((3, 4))
b_np = np.zeros((3, 4))
print(f"  NumPy   : 두 (3, 4) 를")
print(f"            np.concatenate([a, b], axis=0) → {np.concatenate([a_np, b_np], axis=0).shape}    ← 0차원이 3+3=6 으로")
print(f"            np.stack([a, b], axis=0)        → {np.stack([a_np, b_np], axis=0).shape}    ← 새 0차원이 생김")

a_pt = torch.zeros(3, 4)
b_pt = torch.zeros(3, 4)
print(f"  PyTorch : torch.cat([a, b], dim=0)        → {tuple(torch.cat([a_pt, b_pt], dim=0).shape)}")
print(f"            torch.stack([a, b], dim=0)      → {tuple(torch.stack([a_pt, b_pt], dim=0).shape)}")
print("")

# ────────────── (4) 실전 패턴 — 평가 결과 누적 ──────────────
# test 루프에서 각 배치의 예측을 모은 뒤 한 번에 합치는 패턴 (양쪽 동일 아이디어).
print("[4] 평가 결과 누적 패턴")

# NumPy 패턴
preds_np_list = [np.random.randn(4, 10) for _ in range(3)]
all_pred_np = np.concatenate(preds_np_list, axis=0)
print(f"  NumPy   : 각 (4, 10) 3 개 → np.concatenate(..., axis=0) → {all_pred_np.shape}")

# PyTorch 패턴 — classification/test.py 와 동일
preds_pt_list = [torch.randn(4, 10) for _ in range(3)]
all_pred_pt = torch.cat(preds_pt_list, dim=0)
print(f"  PyTorch : 각 (4, 10) 3 개 → torch.cat(..., dim=0)        → {tuple(all_pred_pt.shape)}    ← classification/test.py 패턴")

# ────────────── 비교 정리 ──────────────
# 합치기 (기존 차원)
#     NumPy   : np.concatenate([x, y, ...], axis=k)
#     PyTorch : torch.cat([x, y, ...], dim=k)
# 쌓기 (새 차원)
#     NumPy   : np.stack([x, y, ...], axis=k)
#     PyTorch : torch.stack([x, y, ...], dim=k)
#
# 인자 이름 차이 : axis ↔ dim
# 사용 결정
#     · 같은 의미의 데이터 더 모은다 → concatenate / cat (예: 배치 누적)
#     · 새로운 축이 생긴다           → stack (예: 동영상 프레임을 시간축으로 쌓기)
# 제약
#     · concatenate/cat : 합칠 차원을 제외한 나머지 차원 크기가 동일해야
#     · stack          : 모든 입력 텐서가 동일한 shape 이어야
