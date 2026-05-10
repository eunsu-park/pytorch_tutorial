# 12_dataset_dataloader.py
# Dataset / DataLoader / random_split — 학습 데이터 공급 파이프라인
#
# tensor/14 챕터의 매직 메서드 (__len__, __getitem__) 가 여기서 직접 사용된다.
# tensor/04 챕터의 수동 슬라이싱·셔플이 어떻게 한 줄(DataLoader) 로 자동화되는지 본다.

import torch
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(0)


# ────────────── (1) Dataset 정의 ──────────────
# 실제 학습용 Dataset 의 골격은 이 세 메서드면 충분 :
#   __init__    : 데이터 준비 / 경로 저장
#   __len__     : 전체 샘플 수
#   __getitem__ : i 번째 샘플 반환 (전처리 포함)
print("[1] Dataset 정의")

class ToyDataset(Dataset):
    """무작위 (x, y) 쌍 — 합이 양수면 1, 아니면 0"""

    def __init__(self, num=200):
        self.x = torch.randn(num, 3)
        self.y = (self.x.sum(dim=1, keepdim=True) > 0).float()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = ToyDataset(num=200)
print(f"  len(dataset) = {len(dataset)}")
print(f"  dataset[0]   = {dataset[0]}")
print("")

# ────────────── (2) DataLoader — 배치/셔플 자동화 ──────────────
print("[2] DataLoader")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,                 # 매 epoch 무작위 순서
    num_workers=0,                # 0 = 메인 프로세스, >0 이면 멀티프로세스 로딩
    drop_last=False,              # True 면 자투리 배치 버림
)

for batch_idx, (xb, yb) in enumerate(loader):
    print(f"  batch {batch_idx:>2d} : x.shape={tuple(xb.shape)}, y.shape={tuple(yb.shape)}")
    if batch_idx >= 3:
        break
print(f"  ... (총 {len(loader)} 개 배치)")
print("")

# 04 의 수동 슬라이싱과 비교
#     수동 :  for i in range(N//B):
#                 batch = data[i*B:(i+1)*B]
#             data = data[torch.randperm(N)]   # 셔플
#     →
#     loader :  for xb, yb in loader:    # 셔플·배치·인덱싱이 자동

# ────────────── (3) random_split — train/val 분할 ──────────────
# 학습 시 검증 세트로 모델 성능을 모니터링하기 위해 데이터를 둘로 나눔.
print("[3] random_split — train/val 분할")

n_total = len(dataset)
n_val   = int(n_total * 0.2)      # 20% 검증
n_train = n_total - n_val

train_set, val_set = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42),     # 재현성을 위해 generator 고정
)
print(f"  전체 : {n_total} → train {len(train_set)}, val {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False)
print(f"  train_loader 배치 수 = {len(train_loader)}")
print(f"  val_loader   배치 수 = {len(val_loader)}")
print("")

# ────────────── (4) 실전 Dataset — 파일 로딩 패턴 ──────────────
# classification/pipeline.py 의 CustomDataset 과 같은 패턴 :
#   __init__   : CSV/디렉토리에서 파일 목록 모으기
#   __getitem__: 파일을 읽고 전처리(resize/normalize/tensor 변환) 후 반환
# 무거운 IO 는 __getitem__ 안에서 하므로 num_workers > 0 으로 병렬화하면 학습 속도↑.

# ────────────── 비교 정리 ──────────────
# Dataset                       DataLoader                       random_split
# ─────────────────────────  ──────────────────────────────  ────────────────────────
# __len__, __getitem__ 정의     셔플/배치/멀티프로세스 자동       하나의 dataset 을 두 개로
# 한 샘플의 전처리 위치          학습 루프에서 `for b in loader`   train/val 또는 train/val/test
#
# 학습 루프 표준 :
#   for epoch in range(num_epochs):
#       model.train()
#       for x, y in train_loader: ...        # 학습
#       model.eval()
#       with torch.no_grad():
#           for x, y in val_loader: ...      # 검증
