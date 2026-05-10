# 01_data_pandas.py
# 데이터 준비 ① — CSV → pandas → numpy → torch tensor
#
# 학습 목표
#   학습 데이터가 CSV 파일로 주어졌을 때, 어떤 단계를 거쳐 PyTorch tensor 로 변환되는지 익힌다.
#   pandas / numpy / torch 사이의 dtype·shape·메모리 관계를 한 번에 본다.

import os
import numpy as np
import pandas as pd
import torch

# ────────────── (0) 가짜 CSV 만들기 ──────────────
# 실제 환경에서는 이미 CSV 가 준비되어 있다고 가정. 여기서는 학습용으로 동적으로 생성.
csv_path = "/tmp/_demo_class2.csv"
np.random.seed(0)
N = 100
x_arr = np.linspace(-1, 1, N)
y_arr = 0.9 * x_arr + 0.3 + np.random.normal(0, 0.05, size=N)
pd.DataFrame({"x": x_arr, "y": y_arr}).to_csv(csv_path, index=False)
print(f"[0] 가짜 CSV 생성 : {csv_path} (행 {N})")
print("")

# ────────────── (1) CSV 읽기 — pandas DataFrame ──────────────
print("[1] pandas 로 읽기")

data = pd.read_csv(csv_path)             # CSV 읽기 → DataFrame
print(f"  type(data) = {type(data).__name__}")
print(f"  shape      = {data.shape}      (rows, cols)")
print(f"  columns    = {data.columns.tolist()}")
print(f"  dtypes     :\n{data.dtypes}")
print(data.head())
print("")

# ────────────── (2) 컬럼 선택 — pandas Series ──────────────
print("[2] 컬럼 선택")

x = data["x"]
print(f"  type(x) = {type(x).__name__}    (Series — 1D 처럼 동작하는 pandas 자료형)")
print(f"  shape   = {x.shape}")
print("")

# ────────────── (3) Series → NumPy 배열 ──────────────
print("[3] Series → NumPy")

x_np = x.to_numpy()                       # x.values 도 동일하지만 to_numpy() 가 권장
print(f"  type    = {type(x_np).__name__}")
print(f"  shape   = {x_np.shape}, dtype = {x_np.dtype}")
print("")

# ────────────── (4) NumPy → PyTorch tensor ──────────────
print("[4] NumPy → tensor")

x_pt = torch.from_numpy(x_np)             # 메모리 공유 (CPU). 복사가 필요하면 torch.tensor(x_np)
print(f"  type    = {type(x_pt).__name__}")
print(f"  shape   = {x_pt.shape}, dtype = {x_pt.dtype}")
print("  주의 : torch.from_numpy 는 numpy 와 메모리 공유 — 한쪽 수정이 양쪽에 반영됨")
print("")

# ────────────── (5) 모양 맞추기 — unsqueeze ──────────────
# nn.Linear 같은 레이어는 보통 (batch, features) 모양을 기대.
# 1D (100,) 를 (100, 1) 로 늘려준다.
print("[5] 모양 맞추기")

x_pt = x_pt.unsqueeze(dim=1)              # (100,) → (100, 1)
print(f"  unsqueeze(dim=1) → shape = {tuple(x_pt.shape)}")
print("")

# ────────────── (6) dtype 맞추기 — float32 ──────────────
# pandas / numpy 는 기본 float64, PyTorch nn 모듈은 기본 float32 → 명시적 변환 필요.
print("[6] dtype 맞추기")

x_pt = x_pt.float()                       # = .to(torch.float32)
print(f"  .float() → dtype = {x_pt.dtype}")
print("")

# ────────────── (7) y 도 같은 단계 ──────────────
# 익숙해지면 한 줄로 합칠 수 있다.
y_pt = torch.from_numpy(data["y"].to_numpy()).unsqueeze(dim=1).float()
print(f"[7] y 한 줄 변환 → shape={tuple(y_pt.shape)}, dtype={y_pt.dtype}")

# 정리
os.remove(csv_path)

# ────────────── 정리 ──────────────
# 표준 변환 흐름
#     pd.read_csv → DataFrame → ['col'] → Series
#                              .to_numpy() → ndarray (float64)
#                              torch.from_numpy() → tensor (float64, 메모리 공유)
#                              .unsqueeze(dim=1) → (N,) → (N, 1)
#                              .float() → float32 (PyTorch 기본 dtype 으로)
#
# 한 줄 표기
#     x = torch.from_numpy(df['x'].to_numpy()).unsqueeze(dim=1).float()
#
# 자주 마주치는 문제
#   - dtype mismatch (모델은 float32 인데 입력이 float64)        → .float()
#   - shape mismatch (Linear 는 (N, D) 기대인데 입력이 (N,))     → .unsqueeze(dim=1)
#   - 메모리 공유로 인한 의도치 않은 변경                         → 필요시 torch.tensor(...) 로 복사
