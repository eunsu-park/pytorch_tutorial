# 24_drouput_2.py
# Dropout Layer

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

model = []
model += [nn.Linear(2, 4)]
model += [nn.ReLU()]
model += [nn.Dropout(p=0.5, inplace=True)] # 일반적인 신경망에서는 activation function을 연산 후에 드롭아웃을 적용
model += [nn.Linear(4, 3)]
model += [nn.Sigmoid()]
model = nn.Sequential(*model)
print(model)
print(f"Number of parameters in model: {get_num_params(model)}")


# 드롭아웃은 학습 시에만 적용하고 테스트 시에는 적용하지 않음
model.train() # 모델을 학습 모드로 설정, dropout이 적용됨
inp = torch.randn(32, 2)
print("Before dropout")
print(inp)
print("")
out = model(inp)
print("After dropout")
print(inp)
print("")

model.eval() # 모델을 평가 모드로 설정, dropout이 적용되지 않음
inp = torch.randn(32, 2)
print("Before dropout")
print(inp)
print("")
out = model(inp)
print("After dropout")
print(inp)
print("")
