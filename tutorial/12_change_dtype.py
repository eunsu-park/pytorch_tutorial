# 12_change_dtype.py
# 데이터 타입(dtype) 변경
# 매핑: x.astype(np.dtype) ↔ t.to(torch.dtype) / t.type(torch.dtype)

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── NumPy ──────────────
print("[NumPy]")

array = np.random.randint(0, 10, (3, 3))   # np.random.randint() : 정수 난수 생성 함수
print(array)
print(array.dtype)
print("")

array = array.astype(np.int32)             # array.astype() : dtype 변경, np.int32(array) 도 가능
print(array)
print(array.dtype)
print("")

array = array.astype(np.float64)
print(array)
print(array.dtype)
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

tensor = torch.randint(10, (3, 3))         # torch.randint() : 정수 난수 생성 함수
print(tensor)
print(tensor.dtype)
print("")

tensor = tensor.to(torch.int)              # tensor.to() : dtype/device 변경
print(tensor)
print(tensor.dtype)
print("")

tensor = tensor.to(torch.double)           # torch.double = torch.float64
print(tensor)
print(tensor.dtype)
print("")

# ────────────── 비교 정리 ──────────────
# - dtype 변경
#     NumPy   : x.astype(np.int32)               → 항상 새 배열 반환
#     PyTorch : t.to(torch.int32)                → 같은 dtype 이면 동일 객체 반환(복사 없음)
#               t.type(torch.int32)              → 구식 표기, 동일 동작
# - 자주 쓰는 단축 별칭
#     torch.float = torch.float32
#     torch.double = torch.float64
#     torch.long  = torch.int64
# - tensor.to() 는 dtype 변경뿐 아니라 device 이동도 가능 (다음 챕터 13 참고)
