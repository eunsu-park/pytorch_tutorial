# 04_model_layers.py
# 모델 정의 ① — 기본 레이어 사용 + nn.Sequential
#
# 학습 목표
#   nn.Linear / nn.Conv2d 의 다양한 인자를 직접 지정해보고,
#   여러 레이어를 nn.Sequential 한 줄로 쌓는 패턴을 익힌다.

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) nn.Linear — 다양한 인자 ──────────────
print("[1] nn.Linear")

# 단일 가중치만 가진 가장 단순한 회귀 (y = w * x, bias 없음)
m1 = nn.Linear(in_features=1, out_features=1, bias=False)
print(f"  Linear(1, 1, bias=False) : weight {tuple(m1.weight.shape)}, params={get_num_params(m1)}")

# 다중 회귀 — 3 개 feature 를 1 개 출력으로 (y = w·x + b)
m2 = nn.Linear(in_features=3, out_features=1)
print(f"  Linear(3, 1)             : weight {tuple(m2.weight.shape)}, bias {tuple(m2.bias.shape)}, params={get_num_params(m2)}")

# Hidden layer — 차원을 늘리는 임베딩
m3 = nn.Linear(in_features=3, out_features=5)
print(f"  Linear(3, 5)             : weight {tuple(m3.weight.shape)}, bias {tuple(m3.bias.shape)}, params={get_num_params(m3)}")

# forward 동작 확인
x = torch.randn(10, 3)                        # (배치 10, 입력 3)
print(f"  m2(x{tuple(x.shape)}) → {tuple(m2(x).shape)}     (배치 차원 유지, 입력 3→1)")
print(f"  m3(x{tuple(x.shape)}) → {tuple(m3(x).shape)}     (배치 차원 유지, 입력 3→5)")
print("")

# ────────────── (2) nn.Conv2d — 다양한 인자 ──────────────
print("[2] nn.Conv2d")

# 흑백 이미지의 첫 conv 레이어 — kernel=3 + padding=1 (크기 보존)
c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
print(f"  Conv2d(1, 16, k=3, p=1)        : weight {tuple(c1.weight.shape)}, params={get_num_params(c1)}")

# RGB 이미지의 첫 conv 레이어 — 다운샘플링
c2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
print(f"  Conv2d(3, 64, k=7, s=2, p=3)   : weight {tuple(c2.weight.shape)}, params={get_num_params(c2)}")

# 1x1 conv — 채널 변환만
c3 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1)
print(f"  Conv2d(64, 10, k=1)            : weight {tuple(c3.weight.shape)}, params={get_num_params(c3)}")

# forward 동작 확인
img = torch.randn(8, 1, 28, 28)               # 배치 8, 흑백, 28x28
print(f"  c1(img{tuple(img.shape)}) → {tuple(c1(img).shape)}     (크기 보존)")

img = torch.randn(8, 3, 224, 224)
print(f"  c2(img{tuple(img.shape)}) → {tuple(c2(img).shape)}     (입력 절반 크기로 다운샘플링)")
print("")

# ────────────── (3) nn.Sequential — 한 줄로 모델 정의 ──────────────
# 단순한 순차 모델 (분기/skip 없음) 은 Sequential 로 충분.
print("[3] nn.Sequential")

# 단순한 MLP — 3 → 1 (이진분류용 sigmoid 포함)
mlp = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
print(f"  단순 MLP : params={get_num_params(mlp)}")
print(f"  mlp(x) → {tuple(mlp(torch.randn(10, 3)).shape)}")

# 더 깊은 분류기 — 3 → 16 → 8 → 4 (3-class 분류)
classifier = nn.Sequential(
    nn.Linear(3, 16), nn.ReLU(),
    nn.Linear(16, 8), nn.ReLU(),
    nn.Linear(8, 4),                           # logit 출력 (Softmax 두지 않음)
)
print(f"  깊은 분류기 : params={get_num_params(classifier)}")
print(f"  classifier(x) → {tuple(classifier(torch.randn(10, 3)).shape)}")
print("")

# CNN 백본 — Conv → ReLU → MaxPool 반복
cnn = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28→14
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14→7
    nn.Flatten(),                                                   # (B, 32, 7, 7) → (B, 1568)
    nn.Linear(32 * 7 * 7, 10),
)
print(f"  CNN 분류기 : params={get_num_params(cnn)}")
print(f"  cnn(img(8,1,28,28)) → {tuple(cnn(torch.randn(8, 1, 28, 28)).shape)}")

# ────────────── 정리 ──────────────
# 자주 쓰는 레이어 인자
#   nn.Linear(in_features, out_features, bias=True)
#   nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
#   nn.MaxPool2d(kernel_size, stride=None)              # stride 생략 시 = kernel_size
#   nn.BatchNorm2d(num_features)
#   nn.Dropout(p=0.5)
#   nn.Flatten()                                         # (B, ...) → (B, prod(...))
#
# Sequential 로 만들기 좋은 모델
#   - 단일 흐름의 분류기/회귀 모델
#   - Conv → BN → ReLU → Pool 같은 표준 블록
#
# Sequential 로 안 되는 경우 → 다음 챕터 (05_model_module.py)
#   - skip connection (ResNet)
#   - 다중 입력/출력
#   - 분기 후 합치기 (U-Net 의 cat, multi-task)
