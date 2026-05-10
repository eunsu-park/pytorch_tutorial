# 14_numpy2torch.py
# numpy array → torch tensor 변환
# 앞 챕터들에서 NumPy와 PyTorch를 나란히 비교했다면, 이번 챕터부터(14, 15)는
# 두 라이브러리를 실제로 잇는 변환 함수들을 다룬다.

import numpy as np
import torch

array = np.array([1, 2, 3])
print(array)
print(array.dtype)
print("")

tensor = torch.tensor(array)               # torch.tensor() : 데이터 복사하여 새 tensor 생성 (권장)
print(tensor)
print(tensor.dtype)
print("")

tensor = torch.Tensor(array)               # torch.Tensor() : 항상 float32 로 변환됨 (권장하지 않음)
print(tensor)
print(tensor.dtype)
print("")

tensor = torch.as_tensor(array)            # torch.as_tensor() : 가능하면 메모리 공유, dtype 자동 추론
print(tensor)
print(tensor.dtype)
print("")

tensor = torch.from_numpy(array)           # torch.from_numpy() : 메모리 공유, dtype 자동 추론 (CPU 한정)
print(tensor)
print(tensor.dtype)
print("")

# ────────────── 비교 정리 ──────────────
# - 변환 함수 비교
#     함수                  복사 여부          dtype                권장도
#     torch.tensor(arr)     항상 복사         array의 dtype 유지    ★ 권장 (안전)
#     torch.Tensor(arr)     복사             강제로 float32        △ 비권장 (의도와 다를 수 있음)
#     torch.as_tensor(arr)  가능하면 공유    array의 dtype 유지    ○ 메모리 절약 시
#     torch.from_numpy(arr) 항상 공유        array의 dtype 유지    ○ 메모리 절약 시 (CPU 한정)
# - 메모리 공유의 함정
#     공유된 경우 NumPy 쪽 수정이 PyTorch tensor 에도 반영됨 — 의도치 않은 부작용 주의
