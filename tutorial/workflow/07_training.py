# 07_training.py
# 모델 훈련 — 표준 학습 루프
#
# 학습 목표
#   앞 챕터들(데이터·모델·loss/optim/scheduler)을 모두 합쳐 동작하는 학습 루프 한 벌 작성.
#   외부 epoch 루프 + 내부 batch 루프 + 4단계 학습 + scheduler.step() 의 위치를 정확히 익힌다.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


# ────────────── (1) 가짜 데이터셋 ──────────────
# 실전에서는 03 챕터의 CSV/MNIST Dataset 을 사용.
class ToyDataset(Dataset):
    """y = 0.9 * x + 0.3 + noise, (3,) 입력 → (1,) 출력"""
    def __init__(self, num=2000):
        self.x = torch.randn(num, 3)
        self.y = (self.x.sum(dim=1, keepdim=True) * 0.3 + 0.5
                  + torch.randn(num, 1) * 0.05)
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]


train_loader = DataLoader(ToyDataset(2000), batch_size=64, shuffle=True)


# ────────────── (2) 모델 + criterion + optimizer + scheduler ──────────────
# 04~06 챕터에서 본 패턴을 그대로 사용
class MyModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.layer(x)


model     = MyModel(3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# ────────────── (3) 학습 루프 — 표준 패턴 ──────────────
# 외부 루프 : epoch 단위
# 내부 루프 : batch 단위 — 4단계 (zero_grad → forward → backward → step)
print("[학습 시작]")

# Dropout / BatchNorm 등이 학습 모드로 동작하도록 설정
model.train()

NUM_EPOCHS = 30
for epoch in range(NUM_EPOCHS):
    losses = []
    for inp, tar in train_loader:
        # 4단계 학습
        optimizer.zero_grad()              # 1. 이전 step 의 .grad 를 0 으로 (backward 가 누적하므로 매 iter 필요)
        out  = model(inp)                  # 2. 순전파 — 모델 출력 계산
        loss = criterion(out, tar)         # 3. 손실 계산
        loss.backward()                    # 4-1. 역전파 — 각 파라미터의 .grad 채움
        optimizer.step()                   # 4-2. .grad 를 보고 가중치 갱신 (Adam 규칙으로)
        losses.append(loss.item())

    # 매 epoch 끝에 scheduler.step() — 학습률 갱신
    # StepLR(step_size=10) 이므로 10, 20 epoch 에서 lr 이 절반으로 감소
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        avg_loss = sum(losses) / len(losses)
        cur_lr   = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch+1:>3d} : avg loss = {avg_loss:.4f}, lr = {cur_lr:.5f}")

print("")
print("[학습 종료]")

# ────────────── 정리 ──────────────
# 표준 학습 루프 한 벌
#   model.train()
#   for epoch in range(N):
#       for inp, tar in train_loader:
#           optimizer.zero_grad()
#           out  = model(inp)
#           loss = criterion(out, tar)
#           loss.backward()
#           optimizer.step()
#       scheduler.step()
#
# 자주 묻는 위치 문제
#   - optimizer.zero_grad() 위치 : forward 전 (이전 .grad 클리어). 누적 학습이면 일부러 안 부르기도 함.
#   - scheduler.step() 위치       : epoch 마지막 한 번. (예외 : ReduceLROnPlateau 는 .step(val_loss))
#   - model.train() 호출 시점     : 학습 루프 시작 전 한 번 (또는 검증 후 다시 train() 으로 복원)
#
# 추가하면 좋은 것
#   - 매 epoch 검증 (valid_loader 로 평가) → 다음 챕터 (08_evaluation)
#   - 가장 좋은 val 모델만 저장 (best checkpoint)
#   - tqdm 같은 진행바
#   - tensorboard / wandb 같은 로깅
