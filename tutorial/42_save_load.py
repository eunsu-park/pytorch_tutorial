# 42_save_load.py
# 모델 저장 / 불러오기 — state_dict 기반
#
# 학습 목표
#   학습이 끝난 모델을 디스크에 저장하고 다시 불러오는 방법을 익힌다.
#   classification/train.py / generation/train.py 가 이 패턴을 사용 중이므로 참고.

import os
import torch
import torch.nn as nn

torch.manual_seed(0)

# ────────────── 모델 정의 ──────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ────────────── (1) state_dict 의 정체 ──────────────
print("[1] state_dict — 학습 가능한 파라미터의 dict")

model = SimpleNet()
sd = model.state_dict()
print("키 목록:")
for k, v in sd.items():
    print(f"  {k:<20s}  shape={tuple(v.shape)}")
print("")

# ────────────── (2) 저장 / 불러오기 ──────────────
print("[2] 저장 후 불러오기")

save_path = "/tmp/_demo_simplenet.pt"

# 권장 패턴 : state_dict 만 저장 (파이썬 객체 전체가 아니라 텐서 dict)
torch.save(model.state_dict(), save_path)
print(f"saved → {save_path}, file size = {os.path.getsize(save_path)} bytes")

# 불러오기 : 같은 구조의 모델을 만들고 load_state_dict
new_model = SimpleNet()
new_model.load_state_dict(torch.load(save_path, map_location="cpu"))
print("load_state_dict 완료")

# 두 모델의 출력이 동일한지 확인
x = torch.randn(2, 8)
with torch.no_grad():
    y1 = model(x)
    y2 = new_model(x)
print(f"두 모델의 출력 동일 여부 : {torch.allclose(y1, y2)}")
print("")

# ────────────── (3) optimizer 와 epoch 도 함께 저장 (체크포인트) ──────────────
print("[3] 체크포인트 — 학습 재개를 위한 종합 저장")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 한 step 흉내
optimizer.zero_grad()
loss = model(x).sum()
loss.backward()
optimizer.step()

ckpt_path = "/tmp/_demo_checkpoint.pt"
torch.save({
    "epoch": 5,
    "network": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "loss": loss.item(),
}, ckpt_path)

ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"checkpoint keys : {list(ckpt.keys())}")
print(f"epoch : {ckpt['epoch']}, loss : {ckpt['loss']:.4f}")
print("")

# 정리
os.remove(save_path)
os.remove(ckpt_path)

# ────────────── 비교 정리 ──────────────
# - 권장        : torch.save(model.state_dict(), path) → 텐서만 저장 (안전, 호환성↑)
# - 비권장      : torch.save(model, path) → 클래스 정의 자체를 함께 저장 (코드 변경 시 깨짐)
# - 학습 재개   : {"network": ..., "optimizer": ..., "epoch": ...} 형태의 dict 로 묶어 저장
# - 불러올 때   : 같은 구조의 모델 인스턴스를 먼저 만들고 load_state_dict 호출
# - device 주의 : torch.load(path, map_location=device) 로 cpu/gpu 환경 차이 대응
