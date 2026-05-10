# 31_linear_vs_conv.py
# Linear Layer와 Conv2d Layer의 파라미터 수 비교

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

## 1채널 128x128 입력 영상을 3채널 64x64 영상으로 출력하기 위한 레이어 생성 
## kernel_size : 3으로 고정

# Linear 레이어로 작업 수행
inp = torch.randn(4, 1, 128, 128)
linear = nn.Linear(128*128, 64*64*3, bias=False)
print(linear)
inp = inp.reshape(4, -1) # 1x128x128 -> 1x(128x128)
out_linear = linear(inp) # 1x(128x128) -> 1x(64x64x3)
out_linear = out_linear.reshape(4, 3, 64, 64) # 1x(64x64x3) -> 1x3x64x64
print(out_linear.size())
nb_params_linear = sum(p.numel() for p in linear.parameters())
print(f"Number of parameters in linear: {nb_params_linear}")
print("")

# Conv2d 레이어로 동일한 작업 수행
inp = torch.randn(4, 1, 128, 128)
conv2d = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1, bias=False)
print(conv2d)
out_conv2d = conv2d(inp)
print(out_conv2d.size())
nb_params_conv2d = sum(p.numel() for p in conv2d.parameters())
print(f"Number of parameters in conv2d: {nb_params_conv2d}")
print("")

## 출력 feature map의 갯수를 크게 늘려도 Parameter 수가 크게 늘지 않음
conv2d = nn.Conv2d(1, 1024, kernel_size=3, stride=2, padding=1, bias=False)
print(conv2d)
out_conv2d = conv2d(inp)
print(out_conv2d.size())
nb_params_conv2d = sum(p.numel() for p in conv2d.parameters())
print(f"Number of parameters in conv2d: {nb_params_conv2d}")
print("")
