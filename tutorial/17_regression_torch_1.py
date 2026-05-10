# 17_regression_torch_1.py
# PyTorch 를 이용한 선형 회귀 (autograd + SGD)
#
# 16 과 비교
#   - 16 : w_pred -= lr * 직접 미분식  (사람이 손으로 미분)
#   - 17 : loss.backward() + optimizer.step()  (autograd 가 미분 자동 계산)
# 즉 "기울기 계산" 부분만 PyTorch 가 대신 해주는 단계.

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

def mse(y_pred, y_target):
    """
    평균 제곱 오차(Mean Squared Error, MSE) 계산

    Args:
        y_pred : torch.Tensor, 예측 값
        y_target : torch.Tensor, 타겟 값

    Returns:
        mse : torch.Tensor, 평균 제곱 오차
    """
    return torch.mean(torch.square(y_pred - y_target))

w_pred = torch.zeros(1, requires_grad=True) # torch.zeros() : 0으로 초기화된 tensor 생성
b_pred = torch.zeros(1, requires_grad=True) # requires_grad=True : tensor에 대한 기울기를 계산하도록 설정
y_pred = w_pred * x + b_pred
loss = mse(y_pred, y_target)
print(f"Initial w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")

learning_rate = 0.0002 # 학습률 설정
optimizer = torch.optim.SGD([w_pred, b_pred], lr=learning_rate) # torch.optim.SGD() : SGD optimizer 생성, lr : 학습률, w_pred, b_pred를 optimizer에 등록
print(optimizer)

for epoch in range(10000):
    y_pred = w_pred * x + b_pred # 예측값 계산
    loss = mse(y_pred, y_target) # 손실값 계산
    optimizer.zero_grad() # optimizer의 gradient 초기화
    loss.backward() # 손실값을 사용하여 기울기 계산
    optimizer.step() # optimizer를 사용하여 기울기를 이용하여 파라미터 업데이트

    if epoch % 1000 == 0 : # 1000번째 epoch마다 중간 결과를 출력
        y_pred = w_pred * x + b_pred
        loss = mse(y_pred, y_target)
        print(f"Epoch: {epoch}, w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")

y_pred = w_pred * x + b_pred
loss = mse(y_pred, y_target)
print(f"Final w: {w_pred.item():5.3f}, b: {b_pred.item():5.3f}, loss: {loss.item():5.3f}")
