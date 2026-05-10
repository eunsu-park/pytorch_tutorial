# 28_save_load.py
# 모델 저장 / 불러오기 — state_dict 기반
#
# 표준 패턴 :  텐서만 담은 dict (state_dict) 를 저장 → 같은 구조의 모델에 불러오기.
# classification/train.py / generation/train.py 가 이 패턴을 사용 중.

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

# 권장 패턴 : state_dict 만 저장
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
print(f"두 모델 출력 동일 여부 : {torch.allclose(y1, y2)}")
print("")

# ────────────── (3) 체크포인트 — 학습 재개를 위해 종합 저장 ──────────────
print("[3] 체크포인트")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 한 step 흉내
optimizer.zero_grad()
loss = model(x).sum()
loss.backward()
optimizer.step()

ckpt_path = "/tmp/_demo_checkpoint.pt"
torch.save({
    "epoch":     5,
    "network":   model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "loss":      loss.item(),
}, ckpt_path)

ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"  체크포인트 키 : {list(ckpt.keys())}")
print(f"  epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}")

# 학습을 재개할 때
new_model = SimpleNet()
new_model.load_state_dict(ckpt["network"])
new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
new_optimizer.load_state_dict(ckpt["optimizer"])
print(f"  재개 시점 epoch = {ckpt['epoch']}")
print("")

# 정리
os.remove(save_path)
os.remove(ckpt_path)

# ────────────── 비교 정리 ──────────────
# - 권장        : torch.save(model.state_dict(), path) → 텐서만 저장 (안전, 호환)
# - 비권장      : torch.save(model, path)              → 클래스 정의까지 저장 (코드 변경 시 깨짐)
# - 학습 재개   : {"network": ..., "optimizer": ..., "epoch": ...} 형태로 묶어 저장
# - 불러올 때   : 같은 구조의 모델 인스턴스를 먼저 만들고 load_state_dict 호출
# - device 주의 : torch.load(path, map_location=device) 로 cpu/gpu 환경 차이 대응
#
# best 모델 저장 패턴 (classification/train.py 참고)
#   매 epoch 검증 → val_loss 최저 갱신 시에만 저장 → 학습 종료 후 best 모델로 평가
