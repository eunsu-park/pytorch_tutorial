# 10_cautions_1.py
# 차원 변경 시 주의사항 ① — 메모리 공유
# reshape 결과는 원본과 같은 메모리를 가리킬 수 있어, 한쪽을 수정하면 다른 쪽도 변함

import numpy as np
import torch

# ────────────── NumPy ──────────────
print("[NumPy]")

x = np.arange(4)
print(x)
print("")

y = x.reshape(2, 2)                       # y는 x와 같은 메모리를 참조 (view)
print(y)
print("")

x[0] = 9                                  # x의 값을 변경
print(y)                                  # x의 변경이 y에도 반영됨
print("")

# ────────────── PyTorch ──────────────
print("[PyTorch]")

x = torch.arange(4)
print(x)
print("")

y = x.reshape(2, 2)                       # 동일하게 view 동작
print(y)
print("")

x[0] = 9
print(y)                                  # x의 변경이 y에도 반영됨
print("")

# ────────────── 비교 정리 ──────────────
# - reshape 결과는 가능한 경우 "view"(메모리 공유) 로 만들어짐 — 양쪽 동일
# - 메모리 공유 여부 확인
#     NumPy   : y.base is x   (혹은 np.shares_memory(x, y))
#     PyTorch : y.data_ptr() == x.data_ptr()
# - 의도치 않은 부작용을 막으려면 다음 챕터(11)의 copy / clone 사용
