# 02_data_dataset.py
# 데이터 준비 ② — torch.utils.data.Dataset 상속해 직접 만들기
#
# 학습 목표
#   tensor/14 의 매직 메서드(__init__/__len__/__getitem__) 가
#   PyTorch Dataset 에서 어떻게 활용되는지 직접 작성해 본다.
#   다음 챕터(03)에서 이 Dataset 을 DataLoader 와 결합한다.

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ────────────── (0) 가짜 CSV 만들기 ──────────────
csv_path = "/tmp/_demo_class2.csv"
np.random.seed(0)
N = 100
x_arr = np.linspace(-1, 1, N)
y_arr = 0.9 * x_arr + 0.3 + np.random.normal(0, 0.05, size=N)
pd.DataFrame({"x": x_arr, "y": y_arr}).to_csv(csv_path, index=False)


# ────────────── (1) Dataset 상속 — 세 메서드 구현 ──────────────
# __init__    : 데이터/경로 준비 (한 번만)
# __len__     : 전체 샘플 수      (DataLoader 가 epoch 길이 결정에 사용)
# __getitem__ : i 번째 샘플 반환  (DataLoader 가 인덱스를 던짐)
class CustomDataset(Dataset):
    """간단한 (x, y) 쌍을 CSV 에서 읽어 (1,) 모양 두 텐서로 반환"""

    def __init__(self, csv_file):
        # CSV 한 번 읽고 numpy 배열로 보관 (Series 보다 인덱싱 빠름)
        df = pd.read_csv(csv_file)
        self.x_list = df["x"].to_numpy(dtype=np.float32)
        self.y_list = df["y"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        # idx 번째 (x, y) 를 (1,) 모양 tensor 두 개로 반환
        x = torch.tensor([self.x_list[idx]])           # shape (1,)
        y = torch.tensor([self.y_list[idx]])           # shape (1,)
        return x, y


# ────────────── (2) 인스턴스 동작 확인 ──────────────
print("[1] Dataset 동작 확인")

dataset = CustomDataset(csv_path)
print(f"  len(dataset) = {len(dataset)}     ← __len__ 호출")
print(f"  dataset[0]   = {dataset[0]}       ← __getitem__ 호출")
print(f"  dataset[5]   = {dataset[5]}")
print("")

# ────────────── (3) 영상 데이터셋 패턴 (참고) ──────────────
# 실전에서는 __getitem__ 안에서 파일을 읽고 전처리한다.
# 아래는 형태만 보여주는 의사코드 — 실제 영상 처리는 03 챕터에서 MNIST 로 시연.
print("[2] 영상 Dataset 패턴 (의사코드)")
print("""
  class ImageDataset(Dataset):
      def __init__(self, csv_file):
          df = pd.read_csv(csv_file)
          self.inp_list = df['inp'].tolist()    # 입력 영상 경로
          self.tar_list = df['tar'].tolist()    # 정답 영상 경로

      def __len__(self):
          return len(self.inp_list)

      def __getitem__(self, idx):
          inp = imread(self.inp_list[idx])              # 파일 읽기
          inp = np.expand_dims(inp, 0).astype(np.float32) / 127.5 - 1.0
          inp = torch.from_numpy(inp)                    # → tensor
          tar = ...   # 같은 과정
          return inp, tar
""")

# ────────────── (4) 인덱싱과 반복 ──────────────
# Dataset 은 인덱싱·반복문이 자동 동작 (Sequence 프로토콜)
print("[3] 인덱싱·반복")

for i, (x, y) in enumerate(dataset):
    if i >= 3:
        break
    print(f"  sample {i} : x={x.tolist()}, y={[round(v, 4) for v in y.tolist()]}")

# 정리
os.remove(csv_path)

# ────────────── 정리 ──────────────
# Dataset 의 세 메서드 역할
#   __init__    : 가벼운 준비만 (경로/리스트 저장). 무거운 IO 는 __getitem__ 에서.
#   __len__     : 전체 샘플 수  (DataLoader 의 epoch 길이를 결정)
#   __getitem__ : 한 샘플의 (입력, 정답) 반환 — 전처리도 여기서
#
# 자주 쓰는 패턴
#   - CSV 에서 (입력, 정답) 컬럼을 읽고 __getitem__ 에서 tensor 반환
#   - 파일 경로 리스트를 받아 __getitem__ 에서 read + transform + tensor
#   - 전처리(resize/normalize) 는 __getitem__ 안에서 (병렬 워커가 분산 처리해줌)
#
# 다음 03 챕터 : DataLoader 가 이 Dataset 을 어떻게 배치/셔플/병렬 로딩 하는지.
