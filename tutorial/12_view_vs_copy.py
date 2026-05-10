# 12_view_vs_copy.py
# 메모리 공유(view) vs 복사(copy) — 디버깅 시 자주 마주치는 동작
#
# 핵심
#   reshape / view 결과는 가능하면 원본과 메모리를 공유한다 (view).
#   한쪽을 수정하면 다른 쪽도 변하므로, 원본을 보존하려면 .clone() 으로 복사해야 함.

import torch

# ────────────── (1) 메모리 공유 — reshape 후 한쪽을 바꾸면? ──────────────
print("[1] 메모리 공유 동작")

x = torch.arange(4)
y = x.reshape(2, 2)                  # y 와 x 는 같은 메모리를 가리킴 (view)

print(f"수정 전 :  x = {x.tolist()},  y =\n{y}")
x[0] = 99
print(f"x[0]=99 후 :  x = {x.tolist()},  y =\n{y}    ← y 도 같이 변함")
print("")

# 메모리 공유 여부 확인
a = torch.arange(6)
b = a.reshape(2, 3)
print(f"같은 메모리 여부 (data_ptr 비교) : {a.data_ptr() == b.data_ptr()}")
print("")

# ────────────── (2) 복사 — clone 으로 새 메모리에 ──────────────
print("[2] clone — 깊은 복사")

x = torch.arange(4)
y = x.clone().reshape(2, 2)          # clone 으로 새 메모리 생성

print(f"수정 전 :  x = {x.tolist()},  y =\n{y}")
x[0] = 99
print(f"x[0]=99 후 :  x = {x.tolist()},  y =\n{y}    ← y 는 그대로")
print("")

# ────────────── (3) clone vs detach — 무엇이 다른가 ──────────────
# clone   : 데이터 복사. autograd 그래프는 유지 (grad 가 원본까지 흐름).
# detach  : 데이터 공유. autograd 그래프만 분리.
# 보통 학습 중 안전한 사본이 필요하면 .detach().clone() 둘 다 사용.
print("[3] clone vs detach")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

c = x.clone()
d = x.detach()
print(f"  x.requires_grad        = {x.requires_grad}")
print(f"  clone().requires_grad  = {c.requires_grad}      ← 그래프 유지")
print(f"  detach().requires_grad = {d.requires_grad}     ← 그래프 분리")
print(f"  clone()  data_ptr 같음 : {c.data_ptr() == x.data_ptr()}    ← 데이터 별개")
print(f"  detach() data_ptr 같음 : {d.data_ptr() == x.data_ptr()}     ← 데이터 공유")
print("")

# ────────────── (4) 학습 중 텐서를 기록할 때 ──────────────
# 학습 루프에서 loss/예측을 리스트에 모을 때 그냥 append 하면 그래프까지 따라옴 → 메모리 누수.
# 반드시 .detach() 또는 .item() 으로 그래프를 분리.
print("[4] 학습 루프 주의사항")
print("  losses.append(loss)           ← ✗ 그래프 누수")
print("  losses.append(loss.item())    ← ✓ 파이썬 숫자로 추출 (scalar)")
print("  preds.append(out.detach().cpu())  ← ✓ 그래프 분리 후 보관")

# ────────────── 비교 정리 ──────────────
# - reshape / view 결과는 가능하면 view (메모리 공유)
# - 메모리 공유 확인
#     NumPy   : y.base is x  /  np.shares_memory(x, y)
#     PyTorch : a.data_ptr() == b.data_ptr()
# - 깊은 복사
#     NumPy   : x.copy()
#     PyTorch : x.clone()              ← 그래프 유지
#               x.detach().clone()     ← 그래프 분리 + 복사
# - 학습 중 기록은 .item() 또는 .detach().cpu() 로 그래프 분리 (메모리 누수 방지)
