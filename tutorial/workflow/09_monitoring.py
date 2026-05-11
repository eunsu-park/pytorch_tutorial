# 09_monitoring.py
# 학습 모니터링 — TensorBoard
#
# 학습 목표
#   07_training 의 학습 루프에 TensorBoard 로깅을 끼워 넣어
#   loss/지표/영상/가중치 분포를 실시간 추적하는 표준 패턴을 익힌다.
#
# 의존성 : pip install tensorboard
# 실행   : 학습 종료 후 터미널에서  tensorboard --logdir=./runs
#          그 뒤 브라우저로 http://localhost:6006 접속

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)


# ────────────── (1) SummaryWriter — 로거 인스턴스 ──────────────
# log_dir 은 한 실험을 식별하는 디렉터리. 보통 './runs/<실험이름>' 패턴.
# 같은 폴더에 여러 실험을 두면 TensorBoard 가 한 화면에서 비교해줌.
print("[1] SummaryWriter 생성")

writer = SummaryWriter(log_dir="./runs/demo_monitoring")
print(f"  log_dir = {writer.log_dir}")
print("")


# ────────────── (2) 학습 데이터·모델 (07_training 과 동일 패턴) ──────────────
class ToyDataset(Dataset):
    """y = 0.3 * sum(x) + 0.5 + noise"""
    def __init__(self, num=1000):
        self.x = torch.randn(num, 3)
        self.y = self.x.sum(dim=1, keepdim=True) * 0.3 + 0.5 + torch.randn(num, 1) * 0.05
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]


train_loader = DataLoader(ToyDataset(800), batch_size=64, shuffle=True)
valid_loader = DataLoader(ToyDataset(200), batch_size=64, shuffle=False)


class MyModel(nn.Module):
    def __init__(self, in_features=3):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 16)
        self.layer2 = nn.Linear(16, 1)
        self.act    = nn.ReLU()
    def forward(self, x):
        return self.layer2(self.act(self.layer1(x)))


model     = MyModel(3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)


# ────────────── (3) 학습 루프 + 모니터링 한 줄씩 ──────────────
print("[2] 학습 + 로깅 시작 (5 epoch)")

NUM_EPOCHS = 5
global_step = 0

for epoch in range(NUM_EPOCHS):

    # --- 학습 ---
    model.train()
    train_losses = []
    for inp, tar in train_loader:
        optimizer.zero_grad()
        out  = model(inp)
        loss = criterion(out, tar)
        loss.backward()
        optimizer.step()

        # add_scalar : 단일 스칼라를 step 축에 기록
        writer.add_scalar("Loss/train_iter", loss.item(), global_step)
        train_losses.append(loss.item())
        global_step += 1

    # --- 검증 ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for inp, tar in valid_loader:
            val_losses.append(criterion(model(inp), tar).item())
    train_loss = sum(train_losses) / len(train_losses)
    val_loss   = sum(val_losses)   / len(val_losses)

    # add_scalars : 여러 스칼라를 한 그래프에 (train·val 비교에 표준)
    writer.add_scalars("Loss/epoch", {"train": train_loss, "val": val_loss}, epoch)

    # add_image : (C, H, W) 또는 (N, C, H, W) tensor 를 영상 패널에 기록
    # 본 예제는 회귀라 영상이 없지만, 분류·생성 모델에서는 모델 입력/출력 영상을 여기 넣는다.
    # 형태 시연을 위해 가짜 영상 한 장 추가.
    fake_img = torch.randn(1, 16, 16).clamp(-1, 1) * 0.5 + 0.5     # (1, 16, 16), 값 [0, 1]
    writer.add_image("sample_image", fake_img, epoch)

    # add_histogram : 모델 가중치의 분포를 epoch 축으로 기록
    # 학습이 진행되며 가중치가 어떻게 퍼지는지 추적 가능 — 학습 진단에 유용.
    writer.add_histogram("layer1.weight", model.layer1.weight, epoch)
    writer.add_histogram("layer2.weight", model.layer2.weight, epoch)

    print(f"  epoch {epoch+1} : train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


# ────────────── (4) writer 종료 ──────────────
# close 를 호출해야 버퍼된 이벤트가 모두 디스크로 flush 됨.
writer.close()
print("")
print("[3] 로깅 종료 — writer.close()")


# ────────────── (5) TensorBoard 실행 ──────────────
# 터미널에서 :
#     tensorboard --logdir=./runs
#     # 또는 특정 실험만 :
#     tensorboard --logdir=./runs/demo_monitoring
# 그 뒤 브라우저로 http://localhost:6006 접속.
#
# 보이는 탭
#   SCALARS      : add_scalar / add_scalars 결과 — loss 곡선, train·val 비교
#   IMAGES       : add_image 결과 — epoch 슬라이더로 영상 변화 확인
#   HISTOGRAMS   : add_histogram 결과 — 가중치 분포의 시간 변화
#   DISTRIBUTIONS: HISTOGRAMS 의 다른 시각화 모드


# ────────────── 정리 ──────────────
# 자주 쓰는 API
#   add_scalar(tag, value, step)
#     · loss/accuracy 같은 단일 지표
#   add_scalars(main_tag, {sub_tag: value, ...}, step)
#     · train·val 비교 등 한 그래프에 여러 곡선
#   add_image(tag, img, step)
#     · img 는 (C, H, W) 또는 (N, C, H, W). 값 범위는 보통 [0, 1].
#     · 영상 그리드는 torchvision.utils.make_grid(...) 와 함께 사용
#   add_histogram(tag, values, step)
#     · 가중치/그래디언트 분포 추적
#   add_graph(model, dummy_input)
#     · 모델 구조 시각화 (선택적)
#
# 실험 관리 팁
#   - log_dir 에 실험 이름을 넣어 비교 (./runs/baseline, ./runs/with_dropout 등)
#   - 학습 중 다른 터미널에서 tensorboard 를 띄워두면 실시간 갱신
#   - global_step 은 iter 단위, epoch 축은 epoch 단위 — 헷갈리지 말 것
