# 03_numpy_torch.py
# NumPy ↔ PyTorch 양방향 변환
#
# NumPy 기반 라이브러리(matplotlib, pandas, scikit-image …) 와 PyTorch 모델 사이를
# 매끄럽게 잇는 핵심 챕터.

import numpy as np
import torch

# ────────────── (1) numpy → torch ──────────────
print("[1] numpy → torch")

array = np.array([1, 2, 3])

t1 = torch.tensor(array)         # 항상 복사. dtype 은 array 를 따름.        ★ 권장 (안전)
t2 = torch.Tensor(array)         # 항상 복사. dtype 은 강제로 float32.       △ 비권장
t3 = torch.as_tensor(array)      # 가능하면 메모리 공유. dtype 자동 추론.    ○ 메모리 절약
t4 = torch.from_numpy(array)     # 항상 메모리 공유 (CPU 한정).               ○ 메모리 절약

for name, t in [("torch.tensor", t1), ("torch.Tensor", t2),
                ("torch.as_tensor", t3), ("torch.from_numpy", t4)]:
    print(f"  {name:<18s} → dtype={t.dtype}, value={t.tolist()}")
print("")

# 메모리 공유 동작 — from_numpy 결과는 numpy 쪽 수정에 영향을 받음
arr = np.array([1, 2, 3])
ten = torch.from_numpy(arr)
arr[0] = 99
print(f"numpy 변경 후 from_numpy tensor → {ten.tolist()}    ← 같이 변함 (메모리 공유)")
print("")

# ────────────── (2) torch → numpy ──────────────
# 정석 패턴 :  CPU tensor → tensor.detach().numpy()
#              GPU tensor → tensor.detach().cpu().numpy()
print("[2] torch → numpy")

t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# t.numpy()  ← requires_grad=True 면 RuntimeError. detach 가 필요한 이유.
arr = t.detach().numpy()
print(f"t.detach().numpy() → {arr}, dtype={arr.dtype}")
print("")

# 각 단계의 의미
#   detach() : autograd 그래프와 분리 (그래프가 있으면 numpy 변환 불가)
#   cpu()    : GPU tensor 라면 CPU 로 이동 (GPU 메모리는 numpy 가 못 봄)
#   numpy()  : 메모리를 공유하는 numpy array 반환

# ────────────── (3) 완전히 분리하고 싶다면 ──────────────
# numpy() 결과는 tensor 와 메모리를 공유하므로, 한쪽 수정이 다른 쪽에 반영됨.
# 완전한 사본이 필요하면 .copy() 한 번 더.
t = torch.tensor([1.0, 2.0, 3.0])
arr_shared = t.numpy()
arr_copied = t.numpy().copy()
t[0] = 99.0
print(f"공유 변환 결과 : {arr_shared}    ← tensor 변경이 반영됨")
print(f"복사 변환 결과 : {arr_copied}   ← 영향 없음")
print("")

# ────────────── 비교 정리 ──────────────
# numpy → torch
#     함수                    복사 여부      dtype                권장도
#     torch.tensor(arr)       항상 복사      array dtype 유지     ★ 권장
#     torch.from_numpy(arr)   항상 공유      array dtype 유지     ○ 메모리 절약 (CPU)
#     torch.as_tensor(arr)    가능하면 공유  array dtype 유지     ○ 위와 유사
#     torch.Tensor(arr)       복사           강제 float32         △ 비권장
#
# torch → numpy
#     CPU tensor : tensor.detach().numpy()
#     GPU tensor : tensor.detach().cpu().numpy()
#     완전 분리  : tensor.detach().cpu().numpy().copy()
