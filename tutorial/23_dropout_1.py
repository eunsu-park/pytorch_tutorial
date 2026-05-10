# 23_dropout_1.py
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

dropout = nn.Dropout(p=0.5, inplace=True) # p : 드롭아웃 확률, inplace=True : 입력 자체를 직접 변경
print(dropout)
print(f"Number of parameters in dropout: {get_num_params(dropout)}") # 드롭아웃 레이어는 학습 가능한 파라미터가 없음

inp = torch.randn(32, 16)
print("Before dropout")
print(inp)
print("")
out = dropout(inp)
print("After dropout")
print(inp) # inplace=True이기 때문에 inp 자체가 변경되어 있음
print("")

# 주의 : 본 예제는 inplace=True 의 동작을 보여주기 위한 것.
#        실전에서는 원본 보존이 안전하므로 inplace=False(기본값) 를 권장.
