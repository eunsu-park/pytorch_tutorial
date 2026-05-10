# 04_indexing.py
# 텐서 인덱싱 — 슬라이싱 / boolean mask / fancy indexing / 배치 추출
# NumPy 와 문법이 거의 동일하므로 양쪽을 나란히 보여 차이 (혹은 동일성) 를 확인한다.

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── (1) 기본 슬라이싱 ──────────────
print("[1] 슬라이싱 — NumPy ↔ PyTorch 문법 완전 동일")

# NumPy
x_np = np.arange(20).reshape(4, 5)
print("  NumPy  :")
print(f"    x[0]      = {x_np[0].tolist()}      (0번째 행)")
print(f"    x[:, 0]   = {x_np[:, 0].tolist()}      (0번째 열)")
print(f"    x[1:3]    = {x_np[1:3].tolist()}     (1~2번째 행)")
print(f"    x[:, -1]  = {x_np[:, -1].tolist()}     (마지막 열)")

# PyTorch — 같은 결과
x_pt = torch.arange(20).reshape(4, 5)
print("  PyTorch :")
print(f"    x[0]      = {x_pt[0].tolist()}")
print(f"    x[:, 0]   = {x_pt[:, 0].tolist()}")
print(f"    x[1:3]    = {x_pt[1:3].tolist()}")
print(f"    x[:, -1]  = {x_pt[:, -1].tolist()}")
print("")

# ────────────── (2) Boolean mask ──────────────
# 조건 텐서로 골라내기. shape 가 같으면 True 인 원소만 1D 로 반환.
print("[2] Boolean mask")

# NumPy
x = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
mask = x > 0
print(f"  NumPy   : mask    = {mask.tolist()}")
print(f"            x[mask] = {x[mask].tolist()}     ← 양수만")
x[x < 0] = 0
print(f"            음수→0  = {x.tolist()}")

# PyTorch
x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
mask = x > 0
print(f"  PyTorch : mask    = {mask.tolist()}")
print(f"            x[mask] = {x[mask].tolist()}")
x[x < 0] = 0
print(f"            음수→0  = {x.tolist()}")
print("")

# ────────────── (3) Fancy indexing ──────────────
# 인덱스 텐서로 원하는 위치만 골라옴. 셔플도 같은 패턴.
print("[3] Fancy indexing")

# NumPy
x_np = np.arange(10) * 10
idx_np = np.array([0, 2, 5, 7])
print(f"  NumPy   : x       = {x_np.tolist()}")
print(f"            x[idx]  = {x_np[idx_np].tolist()}     ← 원하는 위치만")

# PyTorch
x_pt = torch.arange(10) * 10
idx_pt = torch.tensor([0, 2, 5, 7])
print(f"  PyTorch : x       = {x_pt.tolist()}")
print(f"            x[idx]  = {x_pt[idx_pt].tolist()}")

# 셔플 — fancy indexing 응용 (양쪽 동일 아이디어)
print(f"  NumPy   : x[np.random.permutation(N)] 로 셔플")
print(f"  PyTorch : x[torch.randperm(N)]        로 셔플")
print("")

# ────────────── (4) 배치 슬라이싱 ──────────────
# 02 챕터의 'for 배치' 패턴을 재현. 양쪽 동일 문법.
print("[4] 배치 슬라이싱 (수동 구현)")

# 양쪽 모두 (1000, 3, 28, 28) 영상 가정
data_np = np.random.randn(1000, 3, 28, 28).astype(np.float32)
data_pt = torch.randn(1000, 3, 28, 28)

batch_size = 128
nb_batch = data_np.shape[0] // batch_size

# 셔플 후 첫 3 개 배치만 출력 (나머지는 동일한 패턴)
data_np = data_np[np.random.permutation(data_np.shape[0])]
data_pt = data_pt[torch.randperm(data_pt.shape[0])]
for batch_idx in range(3):
    s = batch_idx * batch_size
    e = (batch_idx + 1) * batch_size
    print(f"  batch {batch_idx} : NumPy {data_np[s:e].shape}  PyTorch {tuple(data_pt[s:e].shape)}")

print(f"  ... (총 {nb_batch}개 배치)")

# ────────────── 비교 정리 ──────────────
# 슬라이싱        : 문법 완전 동일 — x[a:b], x[:, k] 등
# boolean mask    : 양쪽 동일 — x[조건]  (조건은 같은 shape 의 bool tensor/array)
# fancy indexing  : 양쪽 동일 — x[index_tensor]
# 셔플
#     NumPy   : x[np.random.permutation(N)]   또는 np.random.shuffle(x) (in-place)
#     PyTorch : x[torch.randperm(N)]
# 배치 추출       : 양쪽 동일 슬라이싱. 실전은 DataLoader (26 챕터) 가 자동화.
