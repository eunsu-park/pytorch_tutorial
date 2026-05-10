# 11_cautions_2.py
# 차원 변경 시 주의사항 ② — 데이터 복사
# 매핑: x.copy() ↔ t.clone()

import numpy as np
import torch

# ────────────── NumPy ──────────────
print("[NumPy]")

x = np.arange(4)
print(x)
print("")

y = x.copy().reshape(2, 2)                # x.copy() : 데이터를 복사하여 새로운 메모리에 저장
print(y)
print("")

x[0] = 9                                  # x의 값을 변경
print(y)                                  # y는 별도 메모리이므로 x의 변경이 반영되지 않음
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

x = torch.arange(4)
print(x)
print("")

y = x.clone().reshape(2, 2)               # x.clone() : 데이터를 복사하여 새로운 메모리에 저장
print(y)
print("")

x[0] = 9
print(y)                                  # x의 변경이 y에 반영되지 않음
print("")

# ────────────── 비교 정리 ──────────────
# - 깊은 복사
#     NumPy   : x.copy()
#     PyTorch : x.clone()
# - PyTorch의 clone 은 autograd 그래프도 함께 복제됨
#     · 그래프 분리가 필요하면 x.detach().clone() 사용
# - 단순히 reshape 한 결과를 안전하게 쓰려면
#     NumPy   : x.copy().reshape(...)   또는 np.array(x).reshape(...)
#     PyTorch : x.clone().reshape(...)
