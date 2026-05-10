# 02_dtype_device.py
# 데이터 타입(dtype) 과 위치(device) — 같이 다루는 이유는 둘 다 tensor.to() 로 바꾸기 때문
#
# 매핑 :  x.astype(np.int32) ↔ t.to(torch.int32)
#         (NumPy 에는 device 개념이 없음 — CPU 전용)

import numpy as np
import torch

# ────────────── (1) 기본 dtype 차이 ──────────────
# ★중요 : 정수 기본형은 양쪽 동일(int64), 실수 기본형은 다름.
print("[1] 기본 dtype")

x_int  = np.array([1, 2, 3])
t_int  = torch.tensor([1, 2, 3])
print(f"정수  NumPy={x_int.dtype}  ↔  PyTorch={t_int.dtype}")    # 둘 다 int64

x_flt  = np.array([1., 2., 3.])
t_flt  = torch.tensor([1., 2., 3.])
print(f"실수  NumPy={x_flt.dtype}  ↔  PyTorch={t_flt.dtype}     ← 다름! float64 vs float32")
print("")

# ────────────── (2) dtype 지정 ──────────────
print("[2] dtype 지정")

x = np.array([1, 2, 3], dtype=np.int32)
t = torch.tensor([1, 2, 3], dtype=torch.int32)
print(f"NumPy  dtype=np.int32     → {x.dtype}")
print(f"PyTorch dtype=torch.int32 → {t.dtype}")
print("")

# ────────────── (3) dtype 변경 ──────────────
print("[3] dtype 변경")

x = np.array([1.5, 2.5, 3.5])
t = torch.tensor([1.5, 2.5, 3.5])

print(f"NumPy   x.astype(np.int32) → {x.astype(np.int32)}")
print(f"PyTorch t.to(torch.int32)  → {t.to(torch.int32)}")
# 단축 별칭 :  torch.float = float32, torch.double = float64, torch.long = int64
print(f"PyTorch t.to(torch.long)   → {t.to(torch.long)}")
print("")

# ────────────── (4) device — CPU/GPU ──────────────
# NumPy 는 CPU 전용. PyTorch 는 모든 tensor 에 device 속성이 있어 명시적으로 이동 관리.
print("[4] device")

t = torch.tensor([1, 2, 3])
print(f"기본 device : {t.device}")

# GPU 사용 가능 여부 확인 (CUDA → MPS(Mac) → CPU 순)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"이번 환경에서 사용할 device : {device}")

# 생성 시점에 위치 지정
t = torch.tensor([1, 2, 3], device=device)
print(f"생성 시 device 지정 → {t.device}")

# 이후 이동 (권장 표기 : t.to(device))
t_moved = t.to("cpu")
print(f"t.to('cpu')   → {t_moved.device}")

# ────────────── 비교 정리 ──────────────
# - dtype 확인   : x.dtype ↔ t.dtype
# - dtype 변경   : x.astype(np.int32) ↔ t.to(torch.int32)
# - device 이동  : t.to(device)  (권장)  /  t.cpu() / t.cuda()
# - to() 는 dtype 과 device 를 동시에 바꿀 수도 있음 : t.to(device, dtype=torch.float32)
# - 학습 시 모델과 데이터는 같은 device 위에 있어야 함 (혼합 시 RuntimeError)
