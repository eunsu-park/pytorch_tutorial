# 11_cat_stack.py
# 텐서 합치기 — cat (기존 차원 따라) / stack (새 차원 만들며)
#
# 핵심 차이
#   cat   : 이미 있는 차원의 길이가 늘어남
#   stack : 새 차원이 하나 생김

import torch

torch.manual_seed(0)

# ────────────── (1) cat — 기존 차원 따라 합치기 ──────────────
# 합칠 차원을 제외한 나머지 차원의 크기는 동일해야 함.
print("[1] torch.cat — 기존 차원 따라")

x = torch.randn(1, 32, 32)
y = torch.randn(1, 32, 32)

z0 = torch.cat([x, y], dim=0)        # 0 차원으로 합침 → (2, 32, 32)
z1 = torch.cat([x, y], dim=1)        # 1 차원으로 합침 → (1, 64, 32)
print(f"  x{tuple(x.shape)} + y{tuple(y.shape)}")
print(f"  cat dim=0 → {tuple(z0.shape)}    ← 첫 차원 길이 1+1=2")
print(f"  cat dim=1 → {tuple(z1.shape)}    ← 두번째 차원 길이 32+32=64")
print("")

# 실전 예 : 채널 방향 결합 — Conditional GAN 의 입력 + 정답 쌍 만들기
inp = torch.randn(8, 1, 256, 256)    # 입력 영상
tar = torch.randn(8, 1, 256, 256)    # 정답 영상
pair = torch.cat([inp, tar], dim=1)  # 채널 방향 결합 → (8, 2, 256, 256)
print(f"  pix2pix 입력 쌍 : cat([inp, tar], dim=1) → {tuple(pair.shape)}")
print("")

# ────────────── (2) stack — 새 차원 만들며 합치기 ──────────────
# 합칠 텐서들은 모두 같은 shape 이어야 함.
print("[2] torch.stack — 새 차원 만들며")

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
c = torch.tensor([7., 8., 9.])

s0 = torch.stack([a, b, c], dim=0)   # 새 0 차원 추가 → (3, 3)
s1 = torch.stack([a, b, c], dim=1)   # 새 1 차원 추가 → (3, 3) 단, 의미는 다름
print(f"  세 개의 (3,) 벡터")
print(f"  stack dim=0 → shape {tuple(s0.shape)}\n{s0}")
print(f"  stack dim=1 → shape {tuple(s1.shape)}\n{s1}")
print("")

# ────────────── (3) cat vs stack — 직접 비교 ──────────────
print("[3] cat 과 stack 의 차이")

a = torch.zeros(3, 4)
b = torch.zeros(3, 4)
print(f"  두 (3, 4) 텐서를 합칠 때 :")
print(f"    cat(dim=0)   → {tuple(torch.cat([a, b], dim=0).shape)}    ← 0차원이 3+3=6 으로 길어짐")
print(f"    stack(dim=0) → {tuple(torch.stack([a, b], dim=0).shape)}  ← 새 0차원이 생겨 (2, 3, 4)")
print("")

# ────────────── (4) 실전 패턴 — 평가 결과 누적 ──────────────
# test 루프에서 각 배치의 예측을 모은 뒤 한 번에 합치는 패턴.
print("[4] 평가 결과 누적 패턴")

predictions = []
for batch_idx in range(3):
    pred = torch.randn(4, 10)        # (배치, 클래스 수)
    predictions.append(pred)
all_pred = torch.cat(predictions, dim=0)
print(f"  각 (4, 10) 3 개 → cat(dim=0) → {tuple(all_pred.shape)}    ← classification/test.py 패턴")

# ────────────── 비교 정리 ──────────────
# - 합치기 (기존 차원)
#     NumPy   : np.concatenate([x, y], axis=k)
#     PyTorch : torch.cat([x, y], dim=k)
# - 쌓기 (새 차원)
#     NumPy   : np.stack([x, y], axis=k)
#     PyTorch : torch.stack([x, y], dim=k)
# - 사용 결정
#     · 같은 의미의 데이터 더 모은다 → cat (예: 배치 누적)
#     · 새로운 축이 생긴다           → stack (예: 동영상 프레임을 시간축으로 쌓기)
