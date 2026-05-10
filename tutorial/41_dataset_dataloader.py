# 41_dataset_dataloader.py
# Dataset / DataLoader — 학습 데이터 공급 파이프라인
#
# 학습 목표
#   02_batch.py 에서 손으로 만들었던 배치 처리를 PyTorch 가 제공하는
#   Dataset / DataLoader 로 표준화한다.
#   38_magic_method.py 의 __len__, __getitem__ 가 여기서 직접 사용된다.

import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


# ────────────── (1) 가장 단순한 Dataset ──────────────
print("[1] 직접 만든 Dataset")

class ToyDataset(Dataset):
    """
    무작위 (x, y) 쌍 100 개를 가지는 간단한 Dataset.
    - __init__    : 데이터 준비 / 경로 저장
    - __len__     : 전체 샘플 수 (DataLoader 가 epoch 길이를 결정할 때 사용)
    - __getitem__ : idx 번째 샘플을 반환 (DataLoader 가 인덱스를 던져줌)
    """
    def __init__(self, num=100):
        self.x = torch.randn(num, 3)
        self.y = (self.x.sum(dim=1, keepdim=True) > 0).float()   # 합이 양수면 1

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = ToyDataset(num=100)
print(f"len(dataset) = {len(dataset)}")
print(f"dataset[0]   = {dataset[0]}")
print("")

# ────────────── (2) DataLoader 로 배치 처리 ──────────────
print("[2] DataLoader 로 배치 단위 순회")

loader = DataLoader(
    dataset,
    batch_size=8,          # 한 번에 8개 샘플
    shuffle=True,          # 매 epoch 무작위 순서
    num_workers=0,         # 0 = 메인 프로세스에서 로딩 (>0 이면 멀티프로세스)
    drop_last=False,       # True 면 마지막 자투리 배치를 버림
)

for batch_idx, (xb, yb) in enumerate(loader):
    print(f"  batch {batch_idx:>2d} : x.shape={tuple(xb.shape)}, y.shape={tuple(yb.shape)}")
    if batch_idx >= 4:
        break

print("")

# ────────────── (3) 02_batch.py 와의 대응 ──────────────
# 02_batch.py 의 수동 코드
#   for epoch in range(epochs):
#       for batch_idx in range(nb_batch):
#           batch = data[batch_idx*B : (batch_idx+1)*B]
#       data = data[torch.randperm(N)]   # 셔플
#
# 위와 동일한 동작이 DataLoader 한 줄로 :
#   for xb, yb in loader:
#       ...
# 셔플과 배치 슬라이싱은 DataLoader 가 자동 수행.
#
# 실전 데이터셋 (예: classification/pipeline.py) 에서는 __getitem__ 안에서
#   ① 파일 로딩 → ② 전처리(resize, normalize) → ③ tensor 변환
# 을 한꺼번에 처리하는 패턴이 일반적.

# ────────────── 비교 정리 ──────────────
# - Dataset : 데이터 한 개를 어떻게 가져올지를 정의 (__getitem__) + 전체 크기 (__len__)
# - DataLoader : 셔플, 배치, 멀티프로세스 로딩, drop_last 등을 자동 처리
# - 학습 루프 안에서는 항상 `for batch in loader:` 한 줄로 순회
