# Tutorial

NumPy ↔ PyTorch 비교부터 신경망 학습 인프라까지를 **28 개 챕터**로 단계적으로 학습합니다.
각 챕터는 단독 실행 가능 (`python tutorial/NN_xxx.py`) 하며, 주석 포함 100라인 안팎으로 작성되어 있습니다.

## 실습 환경

- Python 3.x
- 의존성: `numpy`, `torch`, `matplotlib`
- GPU 환경에서만 실행되는 부분(02 의 device 일부)은 메모만 보고 넘어가도 무방

## 8 개 파트 구성

### Part 1 · 텐서 기본 (1 ~ 4)

| # | 파일 | 주제 |
|---:|---|---|
| 01 | `01_tensor_basics.py` | rank/shape, 인덱싱, 생성 함수 (zeros/ones/arange/linspace/eye/randn) |
| 02 | `02_dtype_device.py` | dtype 확인/지정/변경 + CPU/GPU device 이동 |
| 03 | `03_numpy_torch.py` | numpy ↔ torch 양방향 변환 + 메모리 공유 동작 |
| 04 | `04_indexing.py` | 슬라이싱, boolean mask, fancy indexing, 배치 추출 |

### Part 2 · 텐서 연산 (5 ~ 8)

| # | 파일 | 주제 |
|---:|---|---|
| 05 | `05_elementwise_broadcast.py` | element-wise 연산 + broadcasting 규칙 |
| 06 | `06_matmul.py` | 벡터 내적, 행렬 곱, 배치 행렬 곱, Linear 본질 |
| 07 | `07_reduce.py` | sum/mean/max/argmax/std (`dim`, `keepdim`) |
| 08 | `08_reshape.py` | reshape / view / `-1` / contiguous |

### Part 3 · 형태 변환 (9 ~ 12)

| # | 파일 | 주제 |
|---:|---|---|
| 09 | `09_unsqueeze_squeeze.py` | 차원 추가/제거 |
| 10 | `10_permute_transpose.py` | 차원 순서 변경 (`(H,W,C) ↔ (C,H,W)`) |
| 11 | `11_cat_stack.py` | 합치기(cat) / 쌓기(stack) |
| 12 | `12_view_vs_copy.py` | 메모리 공유 vs clone, detach 차이 |

### Part 4 · 자동 미분과 모델 추상화 (13 ~ 14)

| # | 파일 | 주제 |
|---:|---|---|
| 13 | `13_autograd.py` | requires_grad / backward / grad / detach / no_grad |
| 14 | `14_magic_method.py` | nn.Module · Dataset 의 기반이 되는 매직 메서드 |

### Part 5 · 회귀로 학습 흐름 익히기 (15 ~ 16)

| # | 파일 | 주제 |
|---:|---|---|
| 15 | `15_regression_manual.py` | NumPy 수동 vs PyTorch + autograd 비교 |
| 16 | `16_regression_nn.py` | nn.Linear + nn.MSELoss + optimizer (표준 4단계) |

### Part 6 · 신경망 레이어 (17 ~ 22)

| # | 파일 | 주제 |
|---:|---|---|
| 17 | `17_linear.py` | Linear 레이어 + Sequential, 파라미터 공식 |
| 18 | `18_activation.py` | ReLU/Sigmoid/Tanh 곡선 + Sequential 안 사용 |
| 19 | `19_dropout.py` | Dropout + train()/eval() 차이 |
| 20 | `20_conv_basic.py` | Conv2d kernel/channels + 파라미터 공식 |
| 21 | `21_conv_size.py` | padding/stride + 출력 크기 공식 |
| 22 | `22_conv_special.py` | Linear vs Conv, Receptive Field, 1×1 Conv |

### Part 7 · 모델 조립 (23 ~ 25)

| # | 파일 | 주제 |
|---:|---|---|
| 23 | `23_pool_norm.py` | MaxPool2d + BatchNorm2d (γ/β, train/eval) |
| 24 | `24_model_sequential.py` | nn.Sequential 로 분류 모델 만들기 |
| 25 | `25_model_module.py` | nn.Module 상속 + skip connection 예시 |

### Part 8 · 학습 인프라 (26 ~ 28)

| # | 파일 | 주제 |
|---:|---|---|
| 26 | `26_dataset_dataloader.py` | Dataset/DataLoader + random_split |
| 27 | `27_loss_optimizer.py` | 손실 함수 비교 + 옵티마이저(SGD/Momentum/Adam) 비교 |
| 28 | `28_save_load.py` | state_dict 저장/불러오기 / 체크포인트 |

## 핵심 함수 매핑 (NumPy ↔ PyTorch)

| 작업 | NumPy | PyTorch |
|---|---|---|
| Tensor 생성 | `np.array([...])` | `torch.tensor([...])` |
| 영행렬·일행렬 | `np.zeros/ones` | `torch.zeros/ones` |
| 등간격 | `np.arange/linspace` | `torch.arange/linspace` |
| 정규/균일 난수 | `np.random.{randn,rand}` | `torch.{randn,rand}` |
| 차원 수 / 크기 | `x.ndim` / `x.shape` | `t.dim()` / `t.shape` |
| dtype | `x.dtype` / `x.astype(...)` | `t.dtype` / `t.to(torch.dtype)` |
| Element-wise | `a + b`, `a * b` | `a + b`, `a * b` |
| 벡터 내적 (1D) | `np.dot(a, b)` | `torch.dot(a, b)` *(1D 전용)* |
| 행렬 곱 | `np.matmul(A, B)` / `A @ B` | `torch.matmul(A, B)` / `A @ B` |
| 배치 행렬 곱 | `np.matmul` | `torch.matmul` / `torch.bmm` |
| reduce | `x.sum/mean/max/argmax(axis=k)` | `t.sum/mean/max/argmax(dim=k)` |
| reshape | `np.reshape` / `.reshape` | `torch.reshape` / `.reshape` / `.view` |
| 차원 추가 | `np.expand_dims(x, k)` | `t.unsqueeze(k)` |
| 차원 제거 | `np.squeeze(x, k)` | `t.squeeze(k)` |
| 차원 순서 변경 | `np.transpose(x, axes)` | `t.permute(*dims)` |
| 두 차원 교환 | `np.swapaxes(x, a, b)` | `t.transpose(a, b)` |
| 합치기 | `np.concatenate([x,y], axis=k)` | `torch.cat([x,y], dim=k)` |
| 쌓기 | `np.stack([x,y], axis=k)` | `torch.stack([x,y], dim=k)` |
| 깊은 복사 | `x.copy()` | `t.clone()` |
| numpy ↔ torch | — | `torch.from_numpy` / `t.detach().cpu().numpy()` |
| device 이동 | (없음 — CPU 전용) | `t.to(device)` / `t.cpu()` |

## 흔한 주의사항

- **`*` 는 element-wise**: 행렬 곱은 `@` 또는 `matmul` ([06](06_matmul.py))
- **기본 실수 dtype**: NumPy `float64` ↔ PyTorch `float32` ([02](02_dtype_device.py))
- **Broadcasting 차원**: `(N,)` vs `(N, 1)` — 결과가 다르므로 `.shape` 확인 ([05](05_elementwise_broadcast.py))
- **메모리 공유**: reshape 결과 수정이 원본에도 반영 — `.clone()` 필요 ([12](12_view_vs_copy.py))
- **`view` vs `reshape`**: view 는 contiguous 일 때만 ([08](08_reshape.py))
- **Dropout / BN 의 train/eval 차이**: 평가 직전 `model.eval()` 호출 ([19, 23](19_dropout.py))
- **CrossEntropyLoss 입력**: logit 을 받음. 모델 마지막에 `nn.Softmax` 를 두지 않음 ([27](27_loss_optimizer.py))

## 실습 방법

```bash
# 단일 파일 실행
python tutorial/01_tensor_basics.py

# 전체 순차 실행 (zsh)
for f in tutorial/[0-9][0-9]_*.py; do echo "=== $f ==="; python "$f"; done
```
