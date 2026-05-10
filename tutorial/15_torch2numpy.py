# 15_torch2numpy.py
# torch tensor → numpy array 변환
# 14 챕터의 반대 방향. 학습된 결과를 NumPy 기반 후처리/시각화 라이브러리(matplotlib, pandas 등)에 넘길 때 자주 사용.
# NVidia GPU가 없는 경우 cuda 관련 코드에서 에러가 발생함

import numpy as np
import torch

tensor = torch.FloatTensor([1, 2, 3])
print(tensor)
print(tensor.dtype)
print("")

array = tensor.detach().numpy()            # tensor.detach() : autograd 그래프 분리
                                           # tensor.numpy()  : numpy array 변환 (메모리 공유, CPU 한정)
print(array)
print(array.dtype)
print("")

tensor = torch.cuda.FloatTensor([1, 2, 3])
print(tensor)
print(tensor.dtype)
print(tensor.device)
print("")
array = tensor.detach().cpu().numpy()      # GPU tensor → CPU 이동 후 numpy 변환
print(array)
print(array.dtype)
print("")

# ────────────── 비교 정리 ──────────────
# - 변환의 정석 패턴
#     CPU tensor : tensor.detach().numpy()
#     GPU tensor : tensor.detach().cpu().numpy()
# - 각 단계의 의미
#     · detach() : autograd 그래프에서 떼어냄 (그래프가 있으면 numpy 변환 불가)
#     · cpu()    : CPU로 이동 (GPU tensor는 numpy로 직접 변환 불가)
#     · numpy()  : 메모리를 공유하는 numpy array 반환
# - 메모리 공유 주의
#     numpy() 결과는 원본 tensor와 메모리를 공유하므로, 한쪽 수정이 다른 쪽에도 반영됨
#     완전히 분리하려면 tensor.detach().cpu().numpy().copy() 사용
