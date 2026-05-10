# 04_indexing.py
# 텐서 인덱싱 — 슬라이싱 / boolean mask / fancy indexing / 배치 추출
#
# 학습 목표
#   학습 데이터를 배치 단위로 자르거나, 조건에 맞는 원소만 골라 쓰는 패턴을 익힌다.

import torch

torch.manual_seed(0)

# ────────────── (1) 기본 슬라이싱 ──────────────
# NumPy 와 문법 동일.
print("[1] 슬라이싱")

x = torch.arange(20).reshape(4, 5)
print(x)
print(f"x[0]      → {x[0].tolist()}      (0번째 행)")
print(f"x[:, 0]   → {x[:, 0].tolist()}      (0번째 열)")
print(f"x[1:3]    → {x[1:3].tolist()}     (1~2번째 행)")
print(f"x[:, -1]  → {x[:, -1].tolist()}     (마지막 열)")
print("")

# ────────────── (2) Boolean mask ──────────────
# 조건 텐서로 골라내기. shape 가 같으면 True 인 원소만 1D 로 반환.
print("[2] Boolean mask")

x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
mask = x > 0                          # tensor([True, False, True, False, True])
print(f"mask    : {mask.tolist()}")
print(f"x[mask] : {x[mask].tolist()}     ← 양수만")
x[x < 0] = 0                          # 조건에 맞는 위치에 값 대입 (in-place)
print(f"음수→0  : {x.tolist()}")
print("")

# ────────────── (3) Fancy indexing — 인덱스 텐서로 골라내기 ──────────────
print("[3] Fancy indexing")

x = torch.arange(10) * 10
print(f"x         : {x.tolist()}")
idx = torch.tensor([0, 2, 5, 7])
print(f"x[idx]    : {x[idx].tolist()}     ← 원하는 위치만 골라옴")

# 셔플과 동치 — 02 의 'PyTorch 셔플' 패턴과 같음
perm = torch.randperm(len(x))
print(f"x[perm]   : {x[perm].tolist()}    ← 무작위 순열로 셔플")
print("")

# ────────────── (4) 배치 슬라이싱 — 학습 루프 패턴 ──────────────
# 02 챕터에서 본 'for' 루프 배치 처리를 인덱싱으로 재현.
print("[4] 배치 슬라이싱 (수동 구현)")

data = torch.randn(1000, 3, 28, 28)   # 1000개 영상 (3, 28, 28)
batch_size = 128
nb_batch = data.shape[0] // batch_size

# 한 epoch 의 첫 3개 배치만 출력 (나머지는 동일)
perm = torch.randperm(data.shape[0])  # 셔플
data = data[perm]
for batch_idx in range(3):
    s = batch_idx * batch_size
    e = (batch_idx + 1) * batch_size
    batch = data[s:e]
    print(f"  batch {batch_idx} : shape={tuple(batch.shape)}")

print(f"  ... (총 {nb_batch}개 배치)")

# ────────────── 비교 정리 ──────────────
# - 슬라이싱        : x[a:b], x[:, k] 등 NumPy 와 동일
# - boolean mask    : x[조건] 으로 원소 추출 / 값 대입 (조건은 같은 shape 의 bool tensor)
# - fancy indexing  : x[index_tensor] — 인덱스 텐서로 위치를 직접 지정
# - 배치 추출       : data[s:e] (수동) 또는 DataLoader (26 챕터에서 본격 학습)
# - 셔플            : data[torch.randperm(N)]  — fancy indexing 의 응용
