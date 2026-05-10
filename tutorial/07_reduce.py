# 07_reduce.py
# Reduce 연산 — 차원을 줄이는 통계 함수
#
# 학습 루프에서 자주 보는 패턴 :
#   loss.mean()            ← scalar 로 줄이기
#   output.argmax(dim=1)   ← 분류 예측 클래스
#   acc = (pred == label).float().mean()
# 이 챕터에서 dim, keepdim 인자의 의미를 정확히 익힌다.

import torch

torch.manual_seed(0)

# ────────────── (1) 전체 reduce — scalar 로 ──────────────
# dim 을 지정하지 않으면 모든 원소에 대해 reduce → rank-0 tensor.
print("[1] 전체 reduce")

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(f"  x.sum()  = {x.sum().item()}")
print(f"  x.mean() = {x.mean().item()}")
print(f"  x.max()  = {x.max().item()}")
print(f"  x.min()  = {x.min().item()}")
print(f"  x.std()  = {x.std().item():.4f}")
print("")

# ────────────── (2) 특정 차원만 reduce — dim ──────────────
# dim=k 를 지정하면 그 차원이 사라짐.
print("[2] dim 지정")

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])     # shape (2, 3)
print(f"  x{tuple(x.shape)}")
print(f"  x.sum(dim=0) = {x.sum(dim=0).tolist()}     ← 행 방향으로 합 (열별 합), shape={tuple(x.sum(dim=0).shape)}")
print(f"  x.sum(dim=1) = {x.sum(dim=1).tolist()}        ← 열 방향으로 합 (행별 합), shape={tuple(x.sum(dim=1).shape)}")
print("")

# ────────────── (3) keepdim — 차원을 유지 ──────────────
# keepdim=True 는 reduce 한 차원을 1로 남김. broadcasting 과 함께 쓰기 좋음.
print("[3] keepdim")

x = torch.randn(3, 4)
m1 = x.mean(dim=1)                   # shape (3,)
m2 = x.mean(dim=1, keepdim=True)     # shape (3, 1)
print(f"  mean(dim=1)             → shape {tuple(m1.shape)}")
print(f"  mean(dim=1, keepdim=T)  → shape {tuple(m2.shape)}")

# 활용 : 행마다 평균을 빼서 정규화 (broadcasting 활용)
x_centered = x - x.mean(dim=1, keepdim=True)   # (3, 4) - (3, 1) → (3, 4)
print(f"  x - mean(keepdim) shape : {tuple(x_centered.shape)}    ← 각 행의 평균을 0으로")
print("")

# ────────────── (4) argmax / argmin — 위치를 반환 ──────────────
# 분류 모델의 'pred = output.argmax(dim=1)' 패턴.
print("[4] argmax / argmin")

logits = torch.tensor([[2.0, 0.5, -1.0],
                       [0.1, 1.5,  0.3],
                       [0.0, 0.0,  3.0]])
pred = logits.argmax(dim=1)          # 각 행에서 가장 큰 값의 위치
print(f"  logits.argmax(dim=1) = {pred.tolist()}     ← 분류 모델의 예측 클래스")

target = torch.tensor([0, 1, 2])
acc = (pred == target).float().mean().item()
print(f"  accuracy = {acc:.4f}")
print("")

# ────────────── (5) 자주 쓰는 reduce 함수 한눈에 ──────────────
# 모두 dim, keepdim 인자를 받음.
print("[5] 그 외")

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(f"  prod          = {x.prod().item()}      (모든 원소 곱)")
print(f"  norm (p=2)    = {x.norm().item():.4f}")
print(f"  any (조건)    = {(x > 3).any().item()}")
print(f"  all (조건)    = {(x > 0).all().item()}")

# ────────────── 비교 정리 ──────────────
# - dim 미지정     : 모든 원소에 대해 reduce → rank-0 tensor (.item() 으로 파이썬 숫자)
# - dim=k          : 차원 k 가 사라짐 (rank 가 1 줄어듦)
# - keepdim=True   : 차원 k 가 크기 1 로 유지됨 (broadcasting 에 유용)
# - 자주 쓰는 패턴
#     loss.mean()                       — scalar 손실 (학습 표준)
#     output.argmax(dim=1)              — 분류 예측 클래스
#     (pred == label).float().mean()    — accuracy
#     x.std(dim=1, keepdim=True)        — 행별 정규화 (BatchNorm 의 기초)
