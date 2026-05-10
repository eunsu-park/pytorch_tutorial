# 13_device.py
# CPU와 GPU 사용하기 (device 개념)
# NVidia GPU가 없는 경우 GPU 관련 코드에서 에러가 발생할 수 있음
#
# [NumPy 비교 메모]
# NumPy는 CPU 전용 라이브러리이며 'device' 라는 개념이 없음.
# 대규모 신경망 학습은 행렬 연산을 수만~수억 번 반복하므로 GPU 가속이 사실상 필수이며,
# 이 때문에 PyTorch는 NumPy와 달리 모든 tensor에 'device' 속성을 두어
# CPU/GPU 사이의 이동을 명시적으로 관리한다.

import torch

tensor = torch.tensor([1, 2, 3])
print(tensor.device)                       # tensor.device : tensor가 위치한 device 확인

device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부
print(device)

cpu = torch.tensor([1, 2, 3])
print(cpu)

gpu = torch.cuda.FloatTensor([1, 2, 3])    # torch.cuda.FloatTensor() : GPU에 float tensor 생성
print(gpu)

tensor = torch.tensor([1, 2, 3], device=device)   # device= 옵션으로 생성 시점에 위치 지정
print(tensor)

cpu = torch.FloatTensor([1, 2, 3])
print(cpu)

gpu = cpu.cuda()                           # tensor.cuda() : tensor를 GPU로 이동
print(gpu)

gpu2cpu = gpu.cpu()                        # tensor.cpu() : tensor를 CPU로 이동
print(gpu2cpu)

cpu2gpu = cpu.to("cuda")                   # tensor.to() : 임의 device로 이동 (권장 표기)
print(cpu2gpu)

# ────────────── 비교 정리 ──────────────
# - NumPy   : 항상 CPU. device 개념 없음. GPU가 필요하면 CuPy 같은 별도 라이브러리 사용.
# - PyTorch :
#     · 생성 시 위치 지정 : torch.tensor([...], device="cuda")
#     · 이후 이동         : t.to("cuda") / t.cuda()        ← 권장: t.to(device)
#                           t.to("cpu")  / t.cpu()
# - 학습 시 모델과 데이터는 같은 device 위에 있어야 함 (혼합 시 RuntimeError)
