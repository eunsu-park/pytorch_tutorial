# 21_activation_1.py
# Activation functions

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_num_params(model):
    """
    모델(레이어)의 파라미터 수를 계산하는 함수
    
    Args:
        model : torch.nn.Module
    Returns:
        num_params : int
    """
    return sum([p.numel() for p in model.parameters()])

relu = nn.ReLU() # ReLU : y = max(0, x)
print(relu)
print(f"Number of parameters in relu: {get_num_params(relu)}") # ReLU 레이어는 학습 가능한 파라미터가 없음

sigmoid = nn.Sigmoid() # Sigmoid : y = 1 / (1 + exp(-x)), 0 <= y <= 1, 이진 분류 문제에서 출력층에 주로 사용됨
print(sigmoid)
print(f"Number of parameters in sigmoid: {get_num_params(sigmoid)}") # Sigmoid 레이어는 학습 가능한 파라미터가 없음

## -1 <= y <= 1
tanh = nn.Tanh() # Tanh : y = (exp(x) - exp(-x)) / (exp(x) + exp(-x)), -1 <= y <= 1
print(tanh)
print(f"Number of parameters in tanh: {get_num_params(tanh)}") # Tanh 레이어는 학습 가능한 파라미터가 없음

# 활성함수의 곡선 모양을 보기 위해 1D 입력(-5 ~ 5)을 사용
inp = torch.linspace(-5.0, 5.0, 200).view(-1, 1)   # (200, 1)
out_relu = relu(inp)
print(out_relu.size())
out_sigmoid = sigmoid(inp)
print(out_sigmoid.size())
out_tanh = tanh(inp)
print(out_tanh.size())

x = inp.numpy().flatten()
plt.plot(x, out_relu.numpy().flatten(),    label='ReLU',    color='r')
plt.plot(x, out_sigmoid.numpy().flatten(), label='Sigmoid', color='g')
plt.plot(x, out_tanh.numpy().flatten(),    label='Tanh',    color='b')
plt.xlim(-5, 5)
plt.ylim(-1.5, 5.0)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.title("Activation Functions")
plt.show()
