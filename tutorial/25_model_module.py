# 25_model_module.py
# 모델 조립 ② — nn.Module 상속 (분기/조건부 흐름이 가능한 표준 패턴)
#
# Sequential 은 단일 흐름만. 실전 모델 (ResNet 의 skip, U-Net 의 cat 등) 은 nn.Module 을 상속해
# forward 를 직접 작성해야 한다.
# 14 챕터에서 본 매직 메서드 (__init__, __call__) 가 여기서 활용된다.

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


class MyModel(nn.Module):
    """간단한 CNN 분류 모델 (24 의 Sequential 과 같은 동작)"""

    def __init__(self, in_channels, num_classes):
        # 반드시 super().__init__() 호출 — nn.Module 의 내부 등록 시스템 활성
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 레이어를 self.xxx 로 등록하면 .parameters() 에 자동 포함됨
        self.feature_extractor = self._build_feature_extractor()
        self.classifier        = self._build_classifier()

    def _build_feature_extractor(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _build_classifier(self):
        return nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        # 여기서는 단일 흐름이지만, 실전에서는 분기/skip 등을 자유롭게 작성 가능
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)            # (B, C, H, W) → (B, C*H*W) - flatten
        x = self.classifier(x)
        return x


# ────────────── 사용 ──────────────
print("[1] 모델 인스턴스 생성과 forward")

model = MyModel(in_channels=3, num_classes=10)
print(model)                                  # __repr__ 가 트리 형태로 출력
print(f"  파라미터 수 : {get_num_params(model)}")

inp = torch.randn(4, 3, 32, 32)
out = model(inp)                              # __call__ 이 forward 를 자동 실행
print(f"  {tuple(inp.shape)} → {tuple(out.shape)}")
print("")

# ────────────── (2) 분기가 있는 forward — skip connection 예 ──────────────
# Sequential 로는 만들 수 없는 패턴. nn.Module 의 진짜 가치.
print("[2] skip connection 예시")

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU()

    def forward(self, x):
        identity = x                          # skip 경로
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity                  # ★ skip — Sequential 로는 못 만듦
        return self.relu(out)

block = ResBlock(channels=16)
inp = torch.randn(2, 16, 32, 32)
print(f"  ResBlock 입력 {tuple(inp.shape)} → 출력 {tuple(block(inp).shape)}")
print(f"  파라미터 수 : {get_num_params(block)}")
print("")

# ────────────── 비교 정리 ──────────────
# Sequential 로 충분 :
#     · 단일 흐름의 단순한 분류/회귀 모델
# nn.Module 상속 필요 :
#     · skip connection (ResNet)
#     · 다중 입력/출력 (multi-task)
#     · 조건부 분기 (eval 시 다른 경로 등)
#     · U-Net 처럼 down 결과를 up 에서 cat
#
# 핵심 규칙
#   1. super().__init__() 반드시 호출
#   2. 학습 파라미터를 가진 레이어는 self.xxx = ... 로 등록 (그래야 parameters() 에 포함)
#   3. forward(self, x) 를 정의 — 이 함수가 model(x) 호출 시 자동 실행됨 (__call__)
