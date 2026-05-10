# 08_reshape.py
# 차원 변경 — reshape / view / -1 활용 (NumPy ↔ PyTorch)

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) reshape — 같은 데이터를 다른 shape 로 ──────────────
# NumPy 와 PyTorch 모두 reshape 함수/메서드를 제공.
print("[1] reshape")

# NumPy
x = np.arange(12)
print(f"  NumPy   : 원본 shape {x.shape}")
print(f"            x.reshape(3, 4)    : {x.reshape(3, 4).shape}")
print(f"            np.reshape(x, (4,3)) : {np.reshape(x, (4, 3)).shape}")

# PyTorch
t = torch.arange(12)
print(f"  PyTorch : 원본 shape {tuple(t.shape)}")
print(f"            t.reshape(3, 4)        : {tuple(t.reshape(3, 4).shape)}")
print(f"            torch.reshape(t, (4,3)) : {tuple(torch.reshape(t, (4, 3)).shape)}")
print("")

# ────────────── (2) view — PyTorch 전용 ──────────────
# NumPy 에도 .view 메서드가 있지만 'dtype 재해석' 용도로 의미가 완전히 다름.
# reshape 와 같은 의미의 view 는 PyTorch 에만 있음.
print("[2] view (PyTorch 전용)")

t = torch.arange(12)
v = t.view(2, 6)
print(f"  t.view(2, 6) : {tuple(v.shape)}")

# permute 후 비연속 → view 불가
t_perm = torch.arange(12).reshape(3, 4).t()    # transpose → 비연속
print(f"  is_contiguous : {t_perm.is_contiguous()}")
try:
    bad = t_perm.view(-1)
except RuntimeError:
    print("  비연속 tensor.view → RuntimeError")
print(f"  해결 1 : reshape (자동 복사)        → shape {t_perm.reshape(-1).shape}")
print(f"  해결 2 : .contiguous().view(...)     → shape {t_perm.contiguous().view(-1).shape}")
print("")

# ────────────── (3) -1 활용 — 자동 차원 계산 ──────────────
# 양쪽 라이브러리 모두 한 번만 -1 사용 가능.
print("[3] '-1' 활용")

# NumPy
x_np = np.random.randn(2, 3, 4)
print(f"  NumPy   : 원본 {x_np.shape}")
print(f"            reshape(-1)      → {x_np.reshape(-1).shape}      (전체 평탄화)")
print(f"            reshape(2, -1)   → {x_np.reshape(2, -1).shape}")
print(f"            reshape(-1, 4)   → {x_np.reshape(-1, 4).shape}")

# PyTorch
x_pt = torch.from_numpy(x_np)
print(f"  PyTorch : 원본 {tuple(x_pt.shape)}")
print(f"            view(-1)         → {tuple(x_pt.view(-1).shape)}")
print(f"            view(2, -1)      → {tuple(x_pt.view(2, -1).shape)}")
print(f"            view(-1, 4)      → {tuple(x_pt.view(-1, 4).shape)}")
print("")

# 가장 흔한 패턴 : 'batch 유지' flatten
batch_np = np.random.randn(8, 3, 32, 32)
batch_pt = torch.randn(8, 3, 32, 32)
print(f"  NumPy   : batch flatten {batch_np.shape} → {batch_np.reshape(batch_np.shape[0], -1).shape}")
print(f"  PyTorch : batch flatten {tuple(batch_pt.shape)} → {tuple(batch_pt.reshape(batch_pt.size(0), -1).shape)}")
print("  ← Conv → Linear 연결 패턴 (양쪽 동일)")
print("")

# ────────────── (4) flatten 함수 ──────────────
# NumPy 의 flatten / ravel, PyTorch 의 flatten.
print("[4] flatten 함수")

x_np = np.random.randn(2, 3, 4)
x_pt = torch.from_numpy(x_np)
print(f"  NumPy   : x.flatten().shape    = {x_np.flatten().shape}        (항상 1D, 복사)")
print(f"            x.ravel().shape      = {x_np.ravel().shape}        (가능하면 view)")
print(f"  PyTorch : x.flatten().shape         = {tuple(x_pt.flatten().shape)}")
print(f"            x.flatten(start_dim=1).shape = {tuple(x_pt.flatten(start_dim=1).shape)}    ← 첫 차원 유지")

# ────────────── 비교 정리 ──────────────
# 동등 함수
#   NumPy   : np.reshape(x, s) / x.reshape(s) / x.reshape(-1) / x.flatten() / x.ravel()
#   PyTorch : torch.reshape(t, s) / t.reshape(s) / t.view(s) / t.flatten()
#
# reshape vs view (PyTorch 전용 구분)
#   · reshape : 가능하면 view, 아니면 자동 복사. 안전한 선택.
#   · view    : contiguous 일 때만 가능. 비연속이면 .contiguous().view(...) / 또는 reshape.
#
# NumPy 의 .view 는 dtype 재해석용 (의미 다름) — 헷갈리지 말 것.
# '-1' 한 번만 사용 가능 (양쪽 모두). 평탄화는 reshape(-1) / view(-1) / flatten().
