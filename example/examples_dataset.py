# examples_dataset.py
# 실전 패턴 — "이미지 파일 + CSV(경로, 라벨)" 형태의 커스텀 Dataset
#
# 학습 목표
#   tutorial/workflow/02_data_dataset.py 가 (x, y) 숫자쌍을 다뤘다면,
#   여기서는 실전에서 가장 흔한 "디스크의 이미지 파일을 CSV 로 인덱싱"
#   하는 패턴을 처음부터 끝까지 직접 만들어 본다.
#
#   (1) MNIST 를 내려받아 PNG 파일로 저장하고, (파일경로, 라벨) 2-컬럼 CSV 생성
#       — 중간에 실제 이미지가 어떻게 생겼는지 plot 으로 확인
#   (2) 그 CSV 를 읽는 Dataset 작성
#       — 파일은 imageio.imread 로 읽고, 라벨은 onehot 벡터로 변환
#
# 필요 패키지 : torch torchvision imageio matplotlib pandas numpy

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# imageio 2.x : imageio.imread / 3.x : imageio.v3.imread 로 분리됨.
# v2 API 를 명시적으로 가져오면 두 버전 모두에서 imageio.imread 가 동작한다.
import imageio.v2 as imageio
import matplotlib.pyplot as plt

NUM_CLASSES = 10                       # MNIST 숫자 0~9
NUM_SAMPLES = 300                      # 데모용으로 일부만 저장 (실전은 전체)
IMG_DIR = "/tmp/_demo_mnist_png"       # 개별 PNG 가 저장될 폴더
CSV_PATH = "/tmp/_demo_mnist_index.csv"


# ══════════════════════════════════════════════════════════════════
#  (1) MNIST → PNG 파일 + (경로, 라벨) CSV 만들기
# ══════════════════════════════════════════════════════════════════

# ────────────── (1-1) MNIST 내려받기 ──────────────
# transform 없이 받으면 한 샘플이 (PIL.Image, int) 로 나온다.
# 우리가 직접 PNG 로 저장할 것이므로 여기서는 변환을 걸지 않는다.
print("[1] MNIST 내려받기")

try:
    from torchvision.datasets import MNIST
except ImportError:
    print("  (torchvision 미설치 — `pip install torchvision` 후 다시 실행)")
    raise SystemExit

download_root = "/tmp/_demo_MNIST"
os.makedirs(download_root, exist_ok=True)
mnist = MNIST(download_root, train=True, download=True)   # transform 없음
print(f"  전체 학습셋 : {len(mnist)} 장")
print(f"  데모로 저장할 수 : {NUM_SAMPLES} 장")
print("")

# ────────────── (1-2) PNG 로 저장하면서 (경로, 라벨) 목록 수집 ──────────────
# 실전 데이터셋도 보통 이렇게 "디스크의 파일 + 메타 CSV" 형태로 주어진다.
print("[2] PNG 저장 + CSV 작성")

os.makedirs(IMG_DIR, exist_ok=True)

rows = []                                       # (filepath, label) 누적
for i in range(NUM_SAMPLES):
    img, label = mnist[i]                       # img: PIL.Image (28x28), label: int
    filepath = os.path.join(IMG_DIR, f"{i:05d}_label{label}.png")
    img.save(filepath)                          # PIL 이 PNG 로 저장
    rows.append((filepath, label))

# 2-컬럼 CSV : 1열 = 파일 경로, 2열 = 라벨
df = pd.DataFrame(rows, columns=["filepath", "label"])
df.to_csv(CSV_PATH, index=False)
print(f"  PNG 저장 위치 : {IMG_DIR}")
print(f"  CSV 생성      : {CSV_PATH}")
print(f"  CSV 미리보기  :\n{df.head()}")
print(f"  라벨 분포     :\n{df['label'].value_counts().sort_index().to_dict()}")
print("")

# ────────────── (1-3) 실제 이미지가 어떻게 생겼는지 plot ──────────────
# CSV 에 적힌 경로를 그대로 imageio 로 다시 읽어서 3x3 격자로 그려 본다.
# (저장 → 다시 읽기 가 잘 동작하는지 눈으로 검증하는 의미도 있다.)
print("[3] 샘플 이미지 plot (matplotlib 창이 뜸 — 닫으면 계속 진행)")

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for ax, (_, row) in zip(axes.flat, df.head(9).iterrows()):
    arr = imageio.imread(row["filepath"])       # CSV 경로 → numpy (28, 28) uint8
    ax.imshow(arr, cmap="gray")
    ax.set_title(f"label = {row['label']}")
    ax.axis("off")
fig.suptitle("MNIST samples (read back via imageio.imread)")
fig.tight_layout()
plt.show()
print("")


# ══════════════════════════════════════════════════════════════════
#  (2) CSV 를 읽는 커스텀 Dataset — imageio.imread + onehot
# ══════════════════════════════════════════════════════════════════

# ────────────── (2-1) Dataset 정의 ──────────────
# __init__    : CSV 한 번 읽어 경로/라벨 리스트만 보관 (가벼운 준비)
# __len__     : 전체 샘플 수
# __getitem__ : i 번째 — 파일을 imageio.imread 로 읽어 tensor 로,
#               라벨은 onehot 벡터로 변환해서 반환
class MNISTCsvDataset(Dataset):
    """(filepath, label) CSV 를 읽어 (image_tensor, onehot_tensor) 를 돌려주는 Dataset"""

    def __init__(self, csv_file, num_classes=NUM_CLASSES):
        df = pd.read_csv(csv_file)
        self.path_list = df["filepath"].tolist()        # 경로 문자열 리스트
        self.label_list = df["label"].to_numpy(dtype=np.int64)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        # ── 입력 이미지 : 파일 → numpy → tensor ──
        img = imageio.imread(self.path_list[idx])        # (28, 28) uint8
        img = img.astype(np.float32) / 255.0             # [0,255] → [0,1] float32
        img = np.expand_dims(img, axis=0)                # (28,28) → (1,28,28) 채널축 추가
        img = torch.from_numpy(img)                      # numpy → tensor

        # ── 라벨 : 정수 → onehot 벡터 ──
        # 예) label=3, num_classes=10 → [0,0,0,1,0,0,0,0,0,0]
        label = int(self.label_list[idx])
        onehot = np.zeros(self.num_classes, dtype=np.float32)
        onehot[label] = 1.0
        onehot = torch.from_numpy(onehot)                # shape (num_classes,)
        # 참고) torch.nn.functional.one_hot(torch.tensor(label), num_classes)
        #       로도 만들 수 있다 (정수 텐서 → onehot, dtype 은 long).

        return img, onehot


# ────────────── (2-2) 동작 확인 ──────────────
print("[4] Dataset 동작 확인")

dataset = MNISTCsvDataset(CSV_PATH)
print(f"  len(dataset) = {len(dataset)}")

img0, onehot0 = dataset[0]
print(f"  dataset[0] 이미지 : shape={tuple(img0.shape)}, dtype={img0.dtype}, "
      f"min={img0.min():.3f}, max={img0.max():.3f}")
print(f"  dataset[0] 라벨   : {onehot0.tolist()}")
print(f"    → argmax = {int(onehot0.argmax())}  (CSV 라벨 {dataset.label_list[0]} 와 일치)")
print("")

# ────────────── (2-3) DataLoader 로 배치 묶기 ──────────────
# Dataset 만 잘 만들면 DataLoader 가 자동으로 배치/셔플/병렬로딩을 해 준다.
print("[5] DataLoader 로 한 배치 확인")

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
xb, yb = next(iter(loader))
print(f"  이미지 배치 : {tuple(xb.shape)}   (B, C, H, W)")
print(f"  라벨  배치 : {tuple(yb.shape)}   (B, num_classes) ← onehot")
print("")

# ────────────── 정리 ──────────────
# 실전 이미지 Dataset 패턴 요약
#   1. 데이터가 "이미지 파일 폴더 + 메타 CSV(경로, 라벨)" 로 주어진다 (가장 흔함)
#   2. __init__   : CSV 만 읽어 경로/라벨 리스트 보관 (무거운 IO 금지)
#   3. __getitem__: imageio.imread 로 파일 읽기 → float/정규화 → 채널축 → tensor
#   4. 라벨       : 분류 문제에서 정수 라벨을 onehot 으로 변환
#                  (단, nn.CrossEntropyLoss 는 onehot 이 아니라 정수 라벨을 받는다 —
#                   onehot 은 MSE/직접 구현 손실이나 학습용 이해에 주로 사용)
#   5. 나머지(배치/셔플/병렬)는 DataLoader 에 위임
#
# 자주 마주치는 문제
#   - imageio.imread 결과는 (H, W) 또는 (H, W, C) — 모델 입력 (C, H, W) 로 맞추기
#   - uint8 [0,255] → float32 정규화 누락 시 학습 불안정
#   - CSV 의 경로가 상대경로면 실행 위치에 따라 깨짐 → 절대경로 권장

# 데모 파일 정리 (생성물을 직접 열어보고 싶으면 아래 두 줄을 주석 처리)
import shutil
os.remove(CSV_PATH)
shutil.rmtree(IMG_DIR)
