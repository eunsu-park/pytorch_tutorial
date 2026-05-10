# 08_reshape.py
# 차원 변경 — reshape / view + '-1' 활용

import torch

torch.manual_seed(0)

# ────────────── (1) reshape — 같은 데이터를 다른 shape 로 ──────────────
print("[1] reshape")

x = torch.arange(12)
print(f"원본       : shape={tuple(x.shape)}, value={x.tolist()}")

y = x.reshape(3, 4)
print(f"reshape(3,4) :\n{y}")

z = torch.reshape(x, (4, 3))   # 모듈 함수 형태도 가능
print(f"reshape(4,3) :\n{z}")
print("")

# ────────────── (2) view — reshape 와 거의 동일하지만 메모리 연속(contiguous) 필요 ──────────────
print("[2] view")

t = torch.arange(12)
v = t.view(2, 6)               # view 는 메모리가 연속일 때만 가능
print(f"view(2,6) :\n{v}")

# permute 후처럼 메모리가 비연속인 경우 view 가 안 됨
t_perm = torch.arange(12).reshape(3, 4).t()    # transpose → 비연속
print(f"is_contiguous : {t_perm.is_contiguous()}")
try:
    bad = t_perm.view(-1)
except RuntimeError as e:
    print(f"  비연속 tensor.view → RuntimeError")
print(f"  해결책 1 : reshape (자동으로 복사)         → {t_perm.reshape(-1).tolist()}")
print(f"  해결책 2 : .contiguous().view(...)          → {t_perm.contiguous().view(-1).tolist()}")
print("")

# ────────────── (3) -1 활용 — 자동 차원 계산 ──────────────
# 한 번만 사용 가능. 나머지 차원으로부터 자동 계산.
print("[3] '-1' 활용")

x = torch.randn(2, 3, 4)
print(f"원본 shape : {tuple(x.shape)}")
print(f"  view(-1)         → shape {tuple(x.view(-1).shape)}             ← 평탄화 (flatten)")
print(f"  view(2, -1)      → shape {tuple(x.view(2, -1).shape)}           ← 첫 차원 유지, 나머지 평탄화")
print(f"  view(-1, 4)      → shape {tuple(x.view(-1, 4).shape)}           ← 마지막 차원만 4 로 고정")
print(f"  view(6, -1)      → shape {tuple(x.view(6, -1).shape)}            ← 첫 차원 6, 자동 4")
print("")

# 가장 흔한 패턴 : 'batch 유지' flatten
batch = torch.randn(8, 3, 32, 32)         # CNN 출력 같은 4D
flat = batch.reshape(batch.size(0), -1)   # (8, 3*32*32) = (8, 3072)
print(f"  batch flatten : {tuple(batch.shape)} → {tuple(flat.shape)}     ← Conv → Linear 연결 패턴")
print("")

# ────────────── (4) reshape vs view 의 결정 기준 ──────────────
# - view    : 메모리 연속이어야 함. 안전성↓ 속도↑ (복사 안 함)
# - reshape : 가능하면 view, 아니면 자동 복사. 안전성↑ — 보통 reshape 권장.
# - flatten : x.flatten() 도 동일한 결과 (시작/끝 차원 지정 가능)
print("[4] flatten 함수")
x = torch.randn(2, 3, 4)
print(f"  x.flatten()           shape : {tuple(x.flatten().shape)}      (전체 평탄화)")
print(f"  x.flatten(start_dim=1) shape : {tuple(x.flatten(start_dim=1).shape)}    (첫 차원 유지)")

# ────────────── 비교 정리 ──────────────
# - 동등 함수
#     NumPy   : np.reshape(x, s) / x.reshape(s)
#     PyTorch : torch.reshape(t, s) / t.reshape(s) / t.view(s)
# - reshape vs view
#     · reshape : 가능하면 view, 아니면 복사. 안전한 선택.
#     · view    : contiguous 일 때만 가능. 비연속이면 .contiguous().view(...)
# - '-1' 한 번만 사용 가능 (두 개 이상이면 모호함으로 에러)
# - 자주 쓰는 패턴
#     · 평탄화           : x.reshape(-1)  /  x.flatten()
#     · 배치 유지 평탄화  : x.reshape(batch, -1)  /  x.flatten(start_dim=1)
