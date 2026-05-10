# 24_model_sequential.py
# 모델 조립 ① — nn.Sequential 로 분류 모델 만들기
#
# Conv → BN → ReLU → MaxPool 블록을 두 번 쌓고, Flatten → Linear → Linear 로 마무리.
# 단순한 모델은 Sequential 만으로 충분히 표현 가능.

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) 특징 추출부 (Conv 블록) ──────────────
# 입력 (128, 3, 32, 32) → MaxPool 두 번 거쳐 (128, 64, 8, 8) 로 축소
print("[1] 특징 추출부 (CNN)")

inp = torch.randn(128, 3, 32, 32)
feature_extractor = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),    # 3*32*9 + 32 = 896
    nn.BatchNorm2d(32),                             # 2 * 32 = 64
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),          # 32 → 16

    nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 32*64*9 + 64 = 18496
    nn.BatchNorm2d(64),                             # 2 * 64 = 128
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),          # 16 → 8
)
print(feature_extractor)
print(f"  파라미터 수 : {get_num_params(feature_extractor)}")
feat = feature_extractor(inp)
print(f"  입력 {tuple(inp.shape)} → 출력 {tuple(feat.shape)}    ← (B, 64, 8, 8)")
print("")

# ────────────── (2) 분류부 (Flatten → Linear) ──────────────
# Conv 결과를 1D 벡터로 펴서 Linear 에 연결.
# 64 * 8 * 8 = 4096 → Dropout → 4096 → 클래스 수
print("[2] 분류부")

# nn.Flatten() 은 (B, C, H, W) → (B, C*H*W) 로 평탄화 (배치 차원 보존)
classifier = nn.Sequential(
    nn.Flatten(),                                   # (128, 64, 8, 8) → (128, 4096)
    nn.Linear(64 * 8 * 8, 128),                     # 4096*128 + 128
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(128, 10),                             # 128*10 + 10 = 1290
    # 모델 출력은 logit. CrossEntropyLoss 가 내부에서 log_softmax 를 적용하므로
    # 분류 모델 마지막에 Softmax 를 두지 않는다.
)
print(classifier)
print(f"  파라미터 수 : {get_num_params(classifier)}")
out = classifier(feat)
print(f"  특징 {tuple(feat.shape)} → 분류 출력 {tuple(out.shape)}    ← (B, 10) logit")
print("")

# ────────────── (3) 두 부분을 하나로 ──────────────
# 가장 단순하게는 Sequential 안에 Sequential 을 또 넣을 수 있다.
print("[3] 통합 모델")

model = nn.Sequential(feature_extractor, classifier)
print(f"  통합 파라미터 수 : {get_num_params(model)}")

inp = torch.randn(4, 3, 32, 32)
out = model(inp)
print(f"  end-to-end : {tuple(inp.shape)} → {tuple(out.shape)}")

# ────────────── 비교 정리 ──────────────
# - Sequential 은 '입력 → 레이어 1 → 레이어 2 → ... → 출력' 의 단일 흐름만 표현.
# - 분기/skip connection (예: ResNet 의 + 연산, U-Net 의 cat) 은 Sequential 로 못 만듦.
#   → 다음 챕터(25 nn.Module) 에서 forward 를 직접 작성해 표현.
# - nn.Flatten() : 배치 차원 유지하며 나머지 차원을 모두 평탄화. CNN→Linear 연결 표준.
# - 분류 모델의 마지막 Linear 출력에 nn.Softmax 를 두지 않는다 (CrossEntropyLoss 가 내부에서 처리).
