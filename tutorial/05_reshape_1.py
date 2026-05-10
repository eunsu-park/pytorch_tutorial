# 05_reshape_1.py
# 데이터의 차원 변경 (reshape)
# 매핑: np.reshape(x, shape) / x.reshape(shape) ↔ torch.reshape(t, shape) / t.reshape(shape) / t.view(shape)

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

# ────────────── NumPy ──────────────
print("[NumPy]")

array = np.random.randn(2, 2)
print(array)
print(array.shape)
print("")

array = np.reshape(array, (1, 4))         # np.reshape() 사용
print(array)
print(array.shape)
print("")

array = array.reshape(1, 4)               # array.reshape() 도 동일하게 사용 가능
print(array)
print(array.shape)
print("")

array = np.reshape(array, (2, 2))
print(array)
print(array.shape)
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

tensor = torch.randn(2, 2)
print(tensor)
print(tensor.shape)
print("")

tensor = torch.reshape(tensor, (4, 1))    # torch.reshape() 사용
print(tensor)
print(tensor.shape)
print("")

tensor = tensor.reshape(1, 4)             # tensor.reshape() 도 동일하게 사용 가능
print(tensor)
print(tensor.shape)
print("")

tensor = tensor.view(2, 2)                # tensor.view() : reshape 와 유사 (단, torch.view 는 없음)
print(tensor)
print(tensor.shape)
print("")

# ────────────── 비교 정리 ──────────────
# - 동등 함수
#     NumPy   : np.reshape(x, shape) / x.reshape(shape)
#     PyTorch : torch.reshape(t, shape) / t.reshape(shape) / t.view(shape)
# - reshape vs view (PyTorch 한정)
#     · reshape : 가능하면 view, 아니면 복사. 안전한 선택.
#     · view    : 메모리 연속(contiguous)일 때만 가능. 비연속이면 에러 → t.contiguous().view(...) 필요.
# - 모듈 함수 형태(np.reshape / torch.reshape) 와 메서드 형태(.reshape) 는 결과 동일
