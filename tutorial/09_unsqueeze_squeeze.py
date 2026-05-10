# 09_unsqueeze_squeeze.py
# 차원 추가 / 제거 — unsqueeze / squeeze
#
# 자주 쓰이는 곳
#   · 단일 영상 → 배치 차원 추가 : (C, H, W) → (1, C, H, W)
#   · 배치 결과의 채널 차원 제거 : (N, 1, H, W) → (N, H, W)
#   · broadcasting 정렬 : (N,) → (N, 1) 로 만들어 (N, 1) + (1, M) = (N, M)

import torch

torch.manual_seed(0)

# ────────────── (1) 차원 추가 — unsqueeze ──────────────
print("[1] unsqueeze")

x = torch.randn(32, 32)              # (32, 32)
y0 = torch.unsqueeze(x, 0)           # 0번 위치에 차원 추가 → (1, 32, 32)
y2 = x.unsqueeze(2)                  # 메서드 형태도 가능 → (32, 32, 1)
y_neg1 = x.unsqueeze(-1)             # 음수 인덱스 = 끝에서부터 → (32, 32, 1)

print(f"  원본          : {tuple(x.shape)}")
print(f"  unsqueeze(0)  : {tuple(y0.shape)}      ← 배치 차원 추가")
print(f"  unsqueeze(2)  : {tuple(y2.shape)}     ← 채널 차원을 끝에 추가")
print(f"  unsqueeze(-1) : {tuple(y_neg1.shape)}     ← 같은 결과 (음수 인덱스)")
print("")

# 단일 영상 → 배치 차원 추가 (모델에 한 장만 넣을 때)
img = torch.randn(3, 28, 28)
batch = img.unsqueeze(0)             # (1, 3, 28, 28)
print(f"  단일 영상 → 배치 : {tuple(img.shape)} → {tuple(batch.shape)}")
print("")

# ────────────── (2) 차원 제거 — squeeze ──────────────
# 크기가 1 인 차원을 제거. 크기가 1 이 아닌 차원은 무시.
print("[2] squeeze")

y = torch.randn(1, 3, 1, 32, 32)
print(f"  원본              : {tuple(y.shape)}")
print(f"  squeeze()         : {tuple(y.squeeze().shape)}     ← 크기 1인 모든 차원 제거")
print(f"  squeeze(0)        : {tuple(y.squeeze(0).shape)}    ← 0번만 (크기 1이라 제거됨)")
print(f"  squeeze(1)        : {tuple(y.squeeze(1).shape)}  ← 1번 (크기 3이라 무시 — 에러 아님)")
print("")

# 가장 흔한 패턴 : 단일 채널 영상의 채널 차원 제거 후 시각화
out = torch.randn(8, 1, 28, 28)      # 모델 출력 (배치, 채널, H, W)
single = out[0].squeeze(0)           # 첫 영상 → (28, 28) 으로 시각화 가능
print(f"  배치 첫 영상 squeeze : {tuple(out[0].shape)} → {tuple(single.shape)}    ← matplotlib 시각화 가능 형태")

# ────────────── 비교 정리 ──────────────
# 차원 추가
#     NumPy   : np.expand_dims(x, axis=k)
#     PyTorch : torch.unsqueeze(t, k)  /  t.unsqueeze(k)
# 차원 제거
#     NumPy   : np.squeeze(x, axis=k)        (k 생략 시 크기 1인 차원 모두 제거)
#     PyTorch : torch.squeeze(t, k)           (k 생략 시 동일)
# - 음수 인덱스 사용 가능 (-1 = 마지막)
# - squeeze 는 크기가 1 이 아닌 차원에는 영향 없음 (안전)
