# 32_receptive_field.py
# Receptive Field — 출력 한 픽셀이 입력의 몇 픽셀 영역을 보는가?
#
# 핵심 아이디어
#   3x3 커널을 두 번 쌓으면 출력의 한 픽셀이 입력의 5x5 영역을 본다.
#   즉 receptive field 는 같지만, 파라미터 수와 비선형성이 달라진다.
#
# 다이어그램 (1차원 단순화)
#   입력          : - - - - - - - -
#   3x3 conv 1회  :   ◯ ◯ ◯ ◯ ◯ ◯       (한 ◯ 가 입력 3픽셀을 본다)
#   3x3 conv 2회  :       ● ● ● ●         (한 ● 가 ◯ 3개를 보고, ◯ 가 다시 입력 3을 보므로 → 5)
#   결과적으로 ● 한 칸 = 입력 5칸 (= 5x5 conv 1회와 동일)
#
# 파라미터 비교
#   3x3 두 번  :  3*3 + 3*3 = 18  (+ 비선형성 사이에 들어갈 수 있어 표현력↑)
#   5x5 한 번  :  5*5      = 25
#   → 같은 RF 를 더 적은 파라미터로 + 깊이로 표현하는 것이 깊은 네트워크의 장점.

import torch
import torch.nn as nn

def get_num_params(model):
    """
    모델(레이어)의 파라미터 수를 계산하는 함수
    
    Args:
        model : torch.nn.Module
    Returns:
        num_params : int
    """
    return sum([p.numel() for p in model.parameters()])

# model1 : 3x3 커널을 사용한 두 개의 Conv2d Layer -> receptive field가 5x5
inp = torch.randn(1, 1, 256, 256)
model1 = []
model1 += [nn.Conv2d(1, 1, kernel_size=3, bias=False)]
model1 += [nn.Conv2d(1, 1, kernel_size=3, bias=False)]
model1 = nn.Sequential(*model1)
print(model1)
out1 = model1(inp)
print(out1.size())
print(f"Number of parameters in model1: {get_num_params(model1)}")
print("")

# model2 : 5x5 커널을 사용한 한 개의 Conv2d Layer -> receptive field가 5x5
model2 = []
model2 += [nn.Conv2d(1, 1, kernel_size=5, bias=False)]
model2 = nn.Sequential(*model2)
print(model2)
out2 = model2(inp)
print(out2.size())
print(f"Number of parameters in model2: {get_num_params(model2)}")
print("")
