# 05_model_module.py
# 모델 정의 ② — nn.Module 상속 + Sequential 과 조합
#
# 학습 목표
#   분기·skip connection 등 단일 흐름이 아닌 모델을 만들 때 nn.Module 을 상속해 forward 를 작성한다.
#   tensor/14 에서 본 매직 메서드 (__init__, __call__) 가 여기서 그대로 활용된다.

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) 가장 기본적인 nn.Module 상속 ──────────────
# 핵심 규칙
#   1. super().__init__() 반드시 호출 — 부모 nn.Module 의 내부 등록 시스템 활성
#   2. 학습 파라미터를 가진 레이어는 self.xxx = ... 로 등록 (자동 추적됨)
#   3. forward(self, x) 를 정의 — model(x) 호출 시 자동 실행 (__call__)
print("[1] nn.Module 상속 기본")

class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()                        # 부모 초기화 (필수)
        self.layer = nn.Linear(in_features, 1)    # self.* 로 등록 — 자동 추적
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # 직접 호출하지 말 것 — model(x) 로 호출하면 __call__ 을 통해 자동 실행됨
        x = self.layer(x)
        x = self.activation(x)
        return x


my_mlp = MLP(in_features=3)
print(my_mlp)                                     # __repr__ 가 트리 형태로 출력
print(f"  params = {get_num_params(my_mlp)}")
out = my_mlp(torch.randn(10, 3))                  # __call__ → forward
print(f"  forward 출력 shape : {tuple(out.shape)}")
print("")

# ────────────── (2) Sequential 과 Module 의 조합 ──────────────
# 한 모델 안에서 여러 Sequential 블록을 self.* 로 등록하고,
# forward 에서 자유롭게 분기·합치기를 한다.
print("[2] Sequential 블록의 조합 (multi-branch + sum)")

class MyModel(nn.Module):
    """
    같은 입력을 세 갈래로 보내고 (서로 다른 깊이의 sub-network),
    각 출력을 더해 최종 sigmoid 를 적용하는 모델.
    Sequential 만으로는 만들 수 없는 구조 — Module 의 진짜 가치.
    """
    def __init__(self, in_features):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features, 1))
        self.layer2 = nn.Sequential(
            nn.Linear(in_features, 3), nn.ReLU(),
            nn.Linear(3, 1),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features, 3), nn.ReLU(),
            nn.Linear(3, 5), nn.ReLU(),
            nn.Linear(5, 3), nn.ReLU(),
            nn.Linear(3, 1),
        )
        self.last = nn.Sigmoid()

    def forward(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(x)
        y3 = self.layer3(x)
        y  = y1 + y2 + y3                         # 세 출력을 합산
        return self.last(y)


mymodel = MyModel(in_features=3)
print(mymodel)
print(f"  params = {get_num_params(mymodel)}")
out = mymodel(torch.randn(10, 3))
print(f"  forward 출력 shape : {tuple(out.shape)}")
print("")

# ────────────── (3) Skip connection 예시 ──────────────
# ResNet 의 핵심 아이디어 : 입력을 출력에 더해 (identity) 깊은 네트워크에서도 그래디언트가 잘 흐르게.
print("[3] Skip connection")

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU()

    def forward(self, x):
        identity = x                              # skip 경로
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity                      # ★ skip — Sequential 로는 못 만듦
        return self.relu(out)


block = ResBlock(channels=16)
print(f"  ResBlock : params={get_num_params(block)}")
print(f"  forward(8, 16, 32, 32) → {tuple(block(torch.randn(8, 16, 32, 32)).shape)}")

# ────────────── 정리 ──────────────
# nn.Module 상속이 필요한 경우
#   - skip connection (ResNet 의 + 연산)
#   - 다중 입력/출력 (multi-task)
#   - 조건부 흐름 (eval 시 다른 경로 등)
#   - U-Net 처럼 down 결과를 up 에서 cat
#
# 권장 패턴
#   - 학습 파라미터 있는 레이어는 모두 self.xxx 로 등록 (그래야 .parameters() 에 포함)
#   - Sequential 블록을 self.feature_extractor / self.classifier 식으로 묶으면 forward 가 짧아짐
#   - print(model) 로 트리 구조를 확인하는 습관 (등록 누락 발견에 유용)
