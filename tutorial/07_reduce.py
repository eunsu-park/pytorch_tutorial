# 07_reduce.py
# Reduce 연산 — 차원을 줄이는 통계 함수 (NumPy ↔ PyTorch)
#
# ★ 인자 이름 차이
#     NumPy   : axis=k, keepdims=True
#     PyTorch : dim=k,  keepdim=True
#   결과는 양쪽 동일.

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) 전체 reduce — scalar 로 ──────────────
# axis/dim 을 지정하지 않으면 모든 원소에 대해 reduce.
print("[1] 전체 reduce")

x_np = np.array([[1., 2., 3.], [4., 5., 6.]])
x_pt = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

print(f"  NumPy    : sum={x_np.sum()}, mean={x_np.mean()}, max={x_np.max()}, min={x_np.min()}, std={x_np.std():.4f}")
print(f"  PyTorch  : sum={x_pt.sum().item()}, mean={x_pt.mean().item()}, max={x_pt.max().item()}, "
      f"min={x_pt.min().item()}, std={x_pt.std().item():.4f}")
# ⚠️  std 의 기본 보정값(ddof) 이 NumPy=0, PyTorch=1 (unbiased) 로 다를 수 있음.
#     필요하면 NumPy 는 ddof=1, PyTorch 는 unbiased=False 로 맞춤.
print("")

# ────────────── (2) 특정 차원만 reduce ──────────────
print("[2] axis= / dim= 지정")

# NumPy : axis=
print(f"  NumPy   : x.sum(axis=0) = {x_np.sum(axis=0).tolist()}     ← 행 방향 합 (열별), shape={x_np.sum(axis=0).shape}")
print(f"            x.sum(axis=1) = {x_np.sum(axis=1).tolist()}        ← 열 방향 합 (행별), shape={x_np.sum(axis=1).shape}")

# PyTorch : dim=
print(f"  PyTorch : x.sum(dim=0)  = {x_pt.sum(dim=0).tolist()}     ← 같은 결과")
print(f"            x.sum(dim=1)  = {x_pt.sum(dim=1).tolist()}")
print("")

# ────────────── (3) keepdims / keepdim ──────────────
# 줄어든 차원을 1로 남겨 broadcasting 과 같이 쓰기 좋게.
print("[3] keepdims / keepdim")

x_np = np.random.randn(3, 4)
x_pt = torch.from_numpy(x_np)

# NumPy
m_np_no   = x_np.mean(axis=1)                  # (3,)
m_np_keep = x_np.mean(axis=1, keepdims=True)   # (3, 1)
print(f"  NumPy   : mean(axis=1)               → {m_np_no.shape}")
print(f"            mean(axis=1, keepdims=T)    → {m_np_keep.shape}")

# PyTorch
m_pt_no   = x_pt.mean(dim=1)
m_pt_keep = x_pt.mean(dim=1, keepdim=True)
print(f"  PyTorch : mean(dim=1)                → {tuple(m_pt_no.shape)}")
print(f"            mean(dim=1, keepdim=T)      → {tuple(m_pt_keep.shape)}")

# 활용 : 행마다 평균 빼기 (broadcasting)
x_np_centered = x_np - x_np.mean(axis=1, keepdims=True)
x_pt_centered = x_pt - x_pt.mean(dim=1, keepdim=True)
print(f"  행별 평균 제거 결과 동일 : {np.allclose(x_np_centered, x_pt_centered.numpy())}")
print("")

# ────────────── (4) argmax / argmin ──────────────
# 분류 모델의 'pred = output.argmax(dim=1)' 패턴.
print("[4] argmax / argmin — 위치 반환")

# NumPy
logits_np = np.array([[2.0, 0.5, -1.0],
                      [0.1, 1.5,  0.3],
                      [0.0, 0.0,  3.0]])
print(f"  NumPy   : np.argmax(logits, axis=1) = {np.argmax(logits_np, axis=1).tolist()}")

# PyTorch
logits_pt = torch.from_numpy(logits_np)
pred = logits_pt.argmax(dim=1)
print(f"  PyTorch : logits.argmax(dim=1)       = {pred.tolist()}")

target = torch.tensor([0, 1, 2])
acc = (pred == target).float().mean().item()
print(f"  분류 accuracy (PyTorch) = {acc:.4f}")
print("")

# ────────────── (5) 그 외 자주 쓰는 reduce ──────────────
print("[5] 그 외")

x = np.array([[1., 2., 3.], [4., 5., 6.]])
print(f"  NumPy   : prod={x.prod()}, norm={np.linalg.norm(x):.4f}, "
      f"any={(x>3).any()}, all={(x>0).all()}")

t = torch.from_numpy(x)
print(f"  PyTorch : prod={t.prod().item()}, norm={t.norm().item():.4f}, "
      f"any={(t>3).any().item()}, all={(t>0).all().item()}")

# ────────────── 비교 정리 ──────────────
# 인자 이름
#     NumPy   : axis=k, keepdims=True
#     PyTorch : dim=k,  keepdim=True
# 자주 쓰는 reduce
#     sum / mean / max / min / std / var / argmax / argmin / prod / norm
#     (any, all 은 bool tensor 에 대한 reduce)
# 함정
#     · NumPy.std 의 기본 ddof=0 (모분산), PyTorch.std 의 기본 unbiased=True (=ddof=1)
#       → 정확히 같은 값을 원하면 둘 중 하나를 맞춰줘야 함
#     · 학습 루프 패턴
#         loss.mean()                    — scalar 손실
#         output.argmax(dim=1)            — 분류 예측 클래스
#         (pred == label).float().mean()  — accuracy
