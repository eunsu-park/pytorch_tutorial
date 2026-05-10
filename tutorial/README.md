# Tutorial

총 **36 개 챕터**를 세 폴더로 나눠 단계적으로 학습합니다.

```
tutorial/
├── tensor/    (14 챕터)  : 텐서 다루기 + autograd + 매직 메서드
├── workflow/  (8 챕터)   : 데이터→모델→학습→평가 워크플로우 용법 익히기
└── nn/        (14 챕터)  : 신경망 회귀·레이어·모델·학습 인프라
```

각 챕터는 단독 실행 가능하며 (`python tutorial/<folder>/NN_xxx.py`), 주석 포함 100라인 안팎입니다.

학습 순서 권장 : **`tensor/`** → **`workflow/`** → **`nn/`**

## tensor — 텐서 다루기 (14 챕터)

| # | 파일 | 주제 |
|---:|---|---|
| 01 | `01_tensor_basics.py` | rank/shape, 인덱싱, 생성 함수 (NumPy ↔ PyTorch) |
| 02 | `02_dtype_device.py` | dtype 확인/지정/변경 + CPU/GPU device 이동 |
| 03 | `03_numpy_torch.py` | numpy ↔ torch 양방향 변환 + 메모리 공유 동작 |
| 04 | `04_indexing.py` | 슬라이싱, boolean mask, fancy indexing, 배치 추출 |
| 05 | `05_elementwise_broadcast.py` | element-wise 연산 + broadcasting 규칙 |
| 06 | `06_matmul.py` | 벡터 내적, 행렬 곱, 배치 행렬 곱, Linear 본질 |
| 07 | `07_reduce.py` | sum/mean/max/argmax/std (`dim`, `keepdim`) |
| 08 | `08_reshape.py` | reshape / view / `-1` / contiguous |
| 09 | `09_unsqueeze_squeeze.py` | 차원 추가/제거 |
| 10 | `10_permute_transpose.py` | 차원 순서 변경 (`(H,W,C) ↔ (C,H,W)`) |
| 11 | `11_cat_stack.py` | 합치기(cat) / 쌓기(stack) |
| 12 | `12_view_vs_copy.py` | 메모리 공유 vs clone, detach 차이 |
| 13 | `13_autograd.py` | requires_grad / backward / grad / detach / no_grad |
| 14 | `14_magic_method.py` | nn.Module · Dataset 의 기반이 되는 매직 메서드 |

## workflow — 학습 워크플로우 용법 (8 챕터)

`tensor/` 와 `nn/` 사이의 연결 고리. 데이터 준비부터 평가까지 **PyTorch 의 표준 용법**을
한 줄씩 익힌다. `nn/` 으로 본격 진입하기 전에 전체 그림을 잡는 단계.

| # | 파일 | 주제 |
|---:|---|---|
| 01 | `01_data_pandas.py` | CSV → pandas → numpy → tensor 표준 변환 |
| 02 | `02_data_dataset.py` | `torch.utils.data.Dataset` 직접 작성 |
| 03 | `03_data_dataloader.py` | DataLoader + transforms + random_split (MNIST) |
| 04 | `04_model_layers.py` | nn.Linear/Conv2d 다양한 인자 + nn.Sequential |
| 05 | `05_model_module.py` | nn.Module 상속 + 분기·skip connection |
| 06 | `06_loss_optim.py` | 손실 함수 + 옵티마이저 + 스케줄러 (한 묶음) |
| 07 | `07_training.py` | 학습 루프 표준 4단계 + scheduler 위치 |
| 08 | `08_evaluation.py` | 평가 루프 + 회귀/분류 지표 (MSE, accuracy, F1) |

## nn — 신경망 (14 챕터)

| # | 파일 | 주제 |
|---:|---|---|
| 01 | `01_regression_manual.py` | NumPy 수동 vs PyTorch + autograd 비교 |
| 02 | `02_regression_nn.py` | nn.Linear + nn.MSELoss + optimizer (표준 4단계) |
| 03 | `03_linear.py` | Linear 레이어 + Sequential, 파라미터 공식 |
| 04 | `04_activation.py` | ReLU/Sigmoid/Tanh 곡선 + Sequential 안 사용 |
| 05 | `05_dropout.py` | Dropout + train()/eval() 차이 |
| 06 | `06_conv_basic.py` | Conv2d kernel/channels + 파라미터 공식 |
| 07 | `07_conv_size.py` | padding/stride + 출력 크기 공식 |
| 08 | `08_conv_special.py` | Linear vs Conv, Receptive Field, 1×1 Conv |
| 09 | `09_pool_norm.py` | MaxPool2d + BatchNorm2d (γ/β, train/eval) |
| 10 | `10_model_sequential.py` | nn.Sequential 로 분류 모델 만들기 |
| 11 | `11_model_module.py` | nn.Module 상속 + skip connection 예시 |
| 12 | `12_dataset_dataloader.py` | Dataset/DataLoader + random_split |
| 13 | `13_loss_optimizer.py` | 손실 함수 비교 + 옵티마이저(SGD/Momentum/Adam) 비교 |
| 14 | `14_save_load.py` | state_dict 저장/불러오기 / 체크포인트 |

## 핵심 함수 매핑 (NumPy ↔ PyTorch)

`tensor/` 폴더에서 다루는 핵심 매핑입니다.

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
| reduce | `x.sum/mean/argmax(axis=k)` | `t.sum/mean/argmax(dim=k)` |
| reshape | `np.reshape` / `.reshape` | `torch.reshape` / `.reshape` / `.view` |
| 차원 추가 | `np.expand_dims(x, k)` | `t.unsqueeze(k)` |
| 차원 제거 | `np.squeeze(x, k)` | `t.squeeze(k)` |
| 차원 순서 변경 | `np.transpose(x, axes)` | `t.permute(*dims)` |
| 두 차원 교환 | `np.swapaxes(x, a, b)` | `t.transpose(a, b)` |
| 합치기 | `np.concatenate(..., axis=k)` | `torch.cat(..., dim=k)` |
| 쌓기 | `np.stack(..., axis=k)` | `torch.stack(..., dim=k)` |
| 깊은 복사 | `x.copy()` | `t.clone()` |
| numpy ↔ torch | — | `torch.from_numpy` / `t.detach().cpu().numpy()` |
| device 이동 | (없음 — CPU 전용) | `t.to(device)` / `t.cpu()` |

## 흔한 주의사항

- **`*` 는 element-wise**: 행렬 곱은 `@` 또는 `matmul` ([tensor/06](tensor/06_matmul.py))
- **기본 실수 dtype**: NumPy `float64` ↔ PyTorch `float32` ([tensor/02](tensor/02_dtype_device.py))
- **Broadcasting 차원**: `(N,)` vs `(N, 1)` — 결과가 다르므로 `.shape` 확인 ([tensor/05](tensor/05_elementwise_broadcast.py))
- **메모리 공유**: reshape 결과 수정이 원본에도 반영 — `.clone()` 필요 ([tensor/12](tensor/12_view_vs_copy.py))
- **`view` vs `reshape`**: view 는 contiguous 일 때만 ([tensor/08](tensor/08_reshape.py))
- **Dropout / BN 의 train/eval 차이**: 평가 직전 `model.eval()` 호출 ([nn/05](nn/05_dropout.py), [nn/09](nn/09_pool_norm.py))
- **CrossEntropyLoss 입력**: logit 을 받음. 모델 마지막에 `nn.Softmax` 를 두지 않음 ([nn/13](nn/13_loss_optimizer.py))

## 실습 방법

```bash
# 단일 파일 실행
python tutorial/tensor/01_tensor_basics.py
python tutorial/workflow/01_data_pandas.py
python tutorial/nn/01_regression_manual.py

# 폴더별 순차 실행 (zsh)
for f in tutorial/tensor/[0-9][0-9]_*.py;   do echo "=== $f ==="; python "$f"; done
for f in tutorial/workflow/[0-9][0-9]_*.py; do echo "=== $f ==="; python "$f"; done
for f in tutorial/nn/[0-9][0-9]_*.py;       do echo "=== $f ==="; python "$f"; done
```

## 추가 의존성

`workflow/03_data_dataloader.py` 는 MNIST 다운로드를 위해 `torchvision` 이 필요합니다.

```bash
pip install torchvision
```
