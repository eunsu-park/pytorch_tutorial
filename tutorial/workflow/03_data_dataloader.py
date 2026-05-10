# 03_data_dataloader.py
# 데이터 준비 ③ — DataLoader + transforms + random_split
#
# 학습 목표
#   - Dataset (02 챕터) 을 DataLoader 로 감싸 배치 단위로 순회하는 표준 패턴
#   - torchvision 의 transforms 로 영상 전처리 파이프라인 구성
#   - random_split 으로 train/valid/test 분할
#
# 본 챕터부터 torchvision 이 필요 (학습용 표준 라이브러리).

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(0)


# ────────────── (1) 02 챕터의 CustomDataset + DataLoader ──────────────
print("[1] CustomDataset + DataLoader")

# 02 의 가짜 CSV 다시 사용
csv_path = "/tmp/_demo_class2.csv"
np.random.seed(0)
N = 100
pd.DataFrame({
    "x": np.linspace(-1, 1, N),
    "y": 0.9 * np.linspace(-1, 1, N) + 0.3 + np.random.normal(0, 0.05, size=N),
}).to_csv(csv_path, index=False)


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.x_list = df["x"].to_numpy(dtype=np.float32)
        self.y_list = df["y"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return torch.tensor([self.x_list[idx]]), torch.tensor([self.y_list[idx]])


dataset = CustomDataset(csv_path)
loader = DataLoader(
    dataset,
    batch_size=16,        # 한 번에 가져올 샘플 수
    shuffle=True,         # 매 epoch 무작위 순서
    num_workers=0,        # 0 = 메인 프로세스 / >0 이면 멀티프로세스 로딩
    drop_last=False,      # True 면 자투리 배치를 버림
)

# DataLoader 는 iterable — for 루프에서 매 epoch 자동으로 처음부터 다시 순회
for batch_idx, (xb, yb) in enumerate(loader):
    print(f"  batch {batch_idx:>2d} : x.shape={tuple(xb.shape)}, y.shape={tuple(yb.shape)}")
    if batch_idx >= 3:
        break
print(f"  ... (총 {len(loader)} 개 배치)")
print("")

os.remove(csv_path)


# ────────────── (2) torchvision MNIST + transforms ──────────────
# 영상 데이터셋의 표준 처리 파이프라인을 익힌다.
# 처음 실행 시 ~10MB 다운로드. 이미 있으면 즉시 로딩.
print("[2] MNIST + transforms")

try:
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
except ImportError:
    print("  (torchvision 미설치 — `pip install torchvision` 후 다시 실행)")
    raise SystemExit

# transforms.Compose : 여러 전처리를 차례로 적용
#   ToTensor  : PIL → tensor, (H, W, C) → (C, H, W), [0, 255] uint8 → [0, 1] float32
#   Normalize : (x - mean) / std → 픽셀값을 평균 0 부근으로 이동
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),       # MNIST 학습셋의 실제 평균/표준편차
])

download_root = "/tmp/_demo_MNIST"
os.makedirs(download_root, exist_ok=True)

# MNIST(...).train=True/False 로 학습/테스트 분리
full_train = MNIST(download_root, train=True,  transform=mnist_transform, download=True)
test_set   = MNIST(download_root, train=False, transform=mnist_transform, download=True)
print(f"  학습셋 크기 : {len(full_train)}")
print(f"  테스트셋 크기 : {len(test_set)}")

# 한 샘플 형태 확인
img, label = full_train[0]
print(f"  샘플 shape : {tuple(img.shape)}, label : {label}")     # (1, 28, 28), 정수
print("")

# ────────────── (3) random_split — 학습 데이터를 train/valid 로 분할 ──────────────
# MNIST 공식 분할 : 학습 60,000 + 테스트 10,000 (검증 분할 없음)
# → 학습 데이터를 50,000 (train) + 10,000 (valid) 로 직접 분할
print("[3] random_split — train/valid 분할")

train_set, valid_set = random_split(
    full_train,
    lengths=[50_000, 10_000],
    generator=torch.Generator().manual_seed(42),         # 재현성
)
print(f"  분할 결과 : train {len(train_set)} / valid {len(valid_set)} / test {len(test_set)}")
print("")

# ────────────── (4) 세 로더 만들기 ──────────────
print("[4] 세 DataLoader")

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=256, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=0)
print(f"  train_loader 배치 수 : {len(train_loader)}")
print(f"  valid_loader 배치 수 : {len(valid_loader)}")
print(f"  test_loader  배치 수 : {len(test_loader)}")

# 한 배치 형태 확인
xb, yb = next(iter(train_loader))
print(f"  한 배치 : x={tuple(xb.shape)}, y={tuple(yb.shape)}")    # (128, 1, 28, 28), (128,)

# ────────────── 정리 ──────────────
# 표준 학습 파이프라인
#   1. Dataset 정의 (또는 빌트인 사용 : MNIST/CIFAR/ImageFolder 등)
#   2. transforms 로 전처리 파이프라인 구성 (영상은 ToTensor + Normalize 가 표준)
#   3. random_split 로 train/valid/test 분할 (이미 분할된 셋이 있으면 그대로)
#   4. DataLoader 로 batch_size/shuffle/num_workers 지정해 감싸기
#   5. 학습 루프에서 `for x, y in loader` 한 줄로 순회
#
# DataLoader 인자 가이드
#   batch_size  : 메모리·연산량 ↔ 일반화 trade-off (보통 32, 64, 128)
#   shuffle     : train=True / valid·test=False
#   num_workers : 0 (메인) ~ CPU 코어 수. 너무 크면 오버헤드.
#   drop_last   : True 면 자투리 배치 폐기 (BatchNorm 안정성에 유리할 수 있음)
