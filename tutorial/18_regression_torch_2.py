# 18_regression_torch_2.py
# PyTorch NN 을 이용한 선형 회귀 (nn.Linear + nn.MSELoss)
#
# 17 과 비교
#   - 17 : w_pred, b_pred 두 tensor 를 직접 만들고 곱·합으로 모델을 표현
#   - 18 : nn.Linear(1, 1) 한 줄로 동일한 모델을 정의 (내부에 weight, bias 자동 등록)
# 손실 함수도 mse 직접 구현 → nn.MSELoss() 로 교체.

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

def generate_dataset(w_true, b_true, num=200):
    """
    y_true = w_true * x + b_true
    x : 0부터 100까지 200개의 데이터 포인트
    y_target : w_true * x + b_true에 노이즈를 추가한 값

    Args:
        w_true : float, 
        b_true : float
        num : int, 데이터 포인트의 개수

    Returns:
    - x : np.ndarray, shape=(num, 1)
    - y_target : np.ndarray, shape=(num, 1)    
    """
    x = np.linspace(0, 100, num) # shape=(num,)
    x = np.expand_dims(x, axis=1) # shape=(num, 1)
    y_true = w_true * x + b_true
    noise = np.random.normal(0, 5, size=y_true.shape) # np.random.normal() : 정규 분포로부터 랜덤한 수를 추출하는 함수
    y_target = y_true + noise
    x = x.astype(np.float32)
    y_target = y_target.astype(np.float32)
    return x, y_target

w_true, b_true = 0.9, 0.3
x, y_target = generate_dataset(w_true, b_true)
x = torch.from_numpy(x) # numpy array to torch tensor
y_target = torch.from_numpy(y_target) # numpy array to torch tensor
y_true = w_true * x + b_true
print(x.shape, y_target.shape)
print(x.dtype, y_target.dtype)
plt.plot(x, y_target, 'o') # 'o' : 원형 마커
plt.plot(x, y_true, 'r-') # 'r-' : 빨간색 실선
plt.show()

model = torch.nn.Linear(1, 1, bias=True) # 피팅할 함수, y = wx + b
loss_function = torch.nn.MSELoss() # 손실함수
learning_rate = 0.0002 # 학습률 설정
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # torch.optim.SGD() : SGD optimizer 생성, lr : 학습률, model의 파라미터를 optimizer에 등록
print(model)
print(optimizer)
print(loss_function)

y_pred = model(x)
loss = loss_function(y_pred, y_target)
w_pred = model.weight.data 
b_pred = model.bias.data
print(f"Initial w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")

for epoch in range(10000): 
    y_pred = model(x) # 예측값 계산
    loss = loss_function(y_pred, y_target) # 손실값 계산
    optimizer.zero_grad() # optimizer의 gradient 초기화
    loss.backward() # 손실값을 사용하여 기울기 계산
    optimizer.step() # optimizer를 사용하여 파라미터 업데이트

    if epoch % 1000 == 0 :
        y_pred = model(x)
        loss = loss_function(y_pred, y_target)
        w_pred = model.weight.data
        b_pred = model.bias.data
        print(f"Epoch: {epoch}, w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")

y_pred = model(x)
loss = loss_function(y_pred, y_target)
w_pred = model.weight.data
b_pred = model.bias.data
print(f"Final w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")
