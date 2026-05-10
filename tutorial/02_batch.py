# 02_batch.py
# 데이터를 배치(batch) 단위로 불러오는 방법
# 매핑: np.random.randn ↔ torch.randn  /  np.random.shuffle ↔ torch.randperm 인덱싱

import numpy as np
import torch

np.random.seed(0)             # 재현성을 위한 시드 고정
torch.manual_seed(0)

# ────────────── NumPy ──────────────
print("[NumPy]")

# (3, 28, 28) 크기의 데이터가 10000개 있다고 가정
data = np.random.randn(10000, 3, 28, 28)   # np.random.randn() : 정규분포 난수 생성 함수

# 배치를 완전 수동으로 불러오는 방법
batch_1 = data[:128]          # batch_1은 0~127번째 데이터를 불러옴 (128개)
print(batch_1.shape)
batch_2 = data[128:256]       # batch_2는 128~255번째 데이터를 불러옴 (128개)
print(batch_2.shape)

# 배치를 인덱스로 불러오는 방법 (n번째 배치만 불러옴)
batch_idx = 0                 # batch_idx : 불러올 배치의 인덱스
batch_size = 128              # batch_size : 배치의 크기
start = batch_idx * batch_size
end = (batch_idx + 1) * batch_size
batch_n = data[start:end]
print(batch_n.shape)

# 배치를 반복문으로 불러오는 방법 (for문으로 모든 배치 + epoch 개념)
epochs = 2                    # epochs : 학습 횟수 (예시이므로 2회만)
batch_size = 128
shuffle = True
nb_batch = data.shape[0] // batch_size   # 편의상 나머지는 제외

for epoch in range(epochs):
    for batch_idx in range(nb_batch):
        s = batch_idx * batch_size
        e = (batch_idx + 1) * batch_size
        batch = data[s:e]
        # print(epoch, batch_idx, batch.shape)   # 출력량이 많아 주석 처리
    if shuffle:               # 한 epoch 종료 후 데이터를 섞음
        np.random.shuffle(data)               # np.random.shuffle() : in-place 셔플
print(f"NumPy 마지막 배치: epoch={epoch}, batch_idx={batch_idx}, shape={batch.shape}")
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

# 동일한 형태의 데이터를 torch tensor 로 생성
data = torch.randn(10000, 3, 28, 28)       # torch.randn() : 정규분포 난수 생성 함수

# 배치를 완전 수동으로 불러오는 방법
batch_1 = data[:128]
print(batch_1.shape)
batch_2 = data[128:256]
print(batch_2.shape)

# 배치를 인덱스로 불러오는 방법
batch_idx = 0
batch_size = 128
start = batch_idx * batch_size
end = (batch_idx + 1) * batch_size
batch_n = data[start:end]
print(batch_n.shape)

# 배치를 반복문으로 불러오는 방법 + epoch
epochs = 2
batch_size = 128
shuffle = True
nb_batch = data.shape[0] // batch_size

for epoch in range(epochs):
    for batch_idx in range(nb_batch):
        s = batch_idx * batch_size
        e = (batch_idx + 1) * batch_size
        batch = data[s:e]
        # print(epoch, batch_idx, batch.shape)
    if shuffle:               # PyTorch에는 np.random.shuffle 같은 in-place 함수가 없으므로
                              # torch.randperm 으로 무작위 인덱스를 만들어 재정렬
        perm = torch.randperm(data.shape[0])  # torch.randperm() : 0~N-1 의 무작위 순열
        data = data[perm]
print(f"PyTorch 마지막 배치: epoch={epoch}, batch_idx={batch_idx}, shape={batch.shape}")
print("")

# ────────────── 비교 정리 ──────────────
# - 정규분포 난수      : np.random.randn(*shape) ↔ torch.randn(*shape)
# - 슬라이싱 문법      : 양쪽 동일 (data[start:end])
# - 데이터 셔플
#     NumPy   : np.random.shuffle(data)            → 첫 축 기준 in-place 셔플
#     PyTorch : data = data[torch.randperm(N)]     → 인덱싱 기반, 새 tensor 반환
# - 실전에서 PyTorch는 torch.utils.data.DataLoader 로 셔플·배치 처리를 자동화 (후속 챕터 참고)
