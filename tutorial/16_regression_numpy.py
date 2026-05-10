# 16_regression_numpy.py
# NumPy 를 이용한 선형 회귀 — 가장 낮은 단계 (수동 gradient descent)
#
# 학습 흐름
#   16 (지금) : NumPy 만으로 손실/기울기/업데이트를 모두 손으로 작성
#   17        : PyTorch tensor + autograd + optim.SGD 로 갈아끼움
#   18        : nn.Linear + nn.MSELoss + optim 로 한 단계 더 추상화
# 같은 데이터로 같은 회귀 문제를 풀어가며 도구가 어떻게 진화하는지를 본다.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)   # 16, 17, 18 모두 같은 시드 → 결과 비교 가능

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
        y_pred : np.ndarray, 예측 값
        y_target : np.ndarray, 타겟 값   

    Returns:
        mse : float, 평균 제곱 오차
    """
    return np.mean(np.square(y_pred - y_target))

w_pred, b_pred = 0.0, 0.0 # w_pred, b_pred는 0.0으로 초기화
y_pred = w_pred * x + b_pred # 초기 예측값
loss = mse(y_pred, y_target) # 초기 손실값
print(f"Initial w: {w_pred:5.3f}, b: {b_pred:5.3f}, loss: {loss:5.3f}")
learning_rate = 0.0002 # 학습률 설정

for epoch in range(10000):
    y_pred = w_pred * x + b_pred # 예측값
    w_pred -= 2 * learning_rate * ((y_pred - y_target) * x).mean() # w_pred 업데이트
    b_pred -= 2 * learning_rate * (y_pred - y_target).mean() # b_pred 업데이트
    
    if epoch % 1000 == 0 : # 1000번째 epoch마다 중간 결과를 출력
        y_pred = w_pred * x + b_pred
        loss = np.square(y_pred - y_target).mean()
        print(f"Epoch: {epoch}, w: {w_pred:5.3f}, b: {b_pred:5.3f}, loss: {loss:5.3f}")

y_pred = w_pred * x + b_pred
loss = np.square(y_pred - y_target).mean()
print(f"Final w: {w_pred:5.3f}, b: {b_pred:5.3f}, loss: {loss:5.3f}")
