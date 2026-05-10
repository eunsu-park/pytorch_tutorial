# Tutorial

NumPy 와 PyTorch 의 기초 문법을 **나란히 비교**하며 익히는 실습 자료입니다.
01 ~ 15 챕터는 동일 연산을 두 라이브러리로 각각 구현해 차이와 공통점을 직접 확인합니다.
16 챕터부터는 PyTorch 중심으로 신경망 구성요소와 학습 흐름을 다룹니다.

## 실습 환경

- Python 3.x
- 의존성: `numpy`, `torch`, `matplotlib`, `pandas`
- GPU 관련 코드(13, 15)는 NVidia GPU + CUDA 환경에서만 끝까지 실행됩니다

## 챕터 구성

### Part 1. NumPy ↔ PyTorch 비교 (01 ~ 15)

| # | 파일 | 주제 |
|---:|---|---|
| 01 | `01_rank.py` | 차원(rank)과 크기(shape) 확인 |
| 02 | `02_batch.py` | 배치 단위 데이터 슬라이싱 |
| 03 | `03_dtype.py` | 데이터 타입(dtype) 확인·지정 |
| 04 | `04_matrix_ops.py` | 행렬 연산 (element-wise/dot/matmul) + broadcasting |
| 05 | `05_reshape_1.py` | 차원 변경 (reshape) |
| 06 | `06_reshape_2.py` | reshape 시 -1 활용 |
| 07 | `07_unsqueeze.py` | 차원 추가/제거 |
| 08 | `08_permute.py` | 차원 순서 변경 |
| 09 | `09_cat.py` | 차원 기준 합치기 |
| 10 | `10_cautions_1.py` | reshape 시 메모리 공유 주의 |
| 11 | `11_cautions_2.py` | 데이터 복사로 메모리 분리 |
| 12 | `12_change_dtype.py` | dtype 변경 |
| 13 | `13_device.py` | CPU/GPU device 개념 (PyTorch 고유) |
| 14 | `14_numpy2torch.py` | numpy → torch 변환 |
| 15 | `15_torch2numpy.py` | torch → numpy 변환 |

### Part 2. 회귀 학습 흐름 (16 ~ 18)

같은 회귀 문제를 도구를 바꿔가며 풀어 PyTorch 의 추상화 단계를 본다.

| # | 파일 | 주제 |
|---:|---|---|
| 16 | `16_regression_numpy.py` | NumPy + 수동 gradient descent |
| 17 | `17_regression_torch_1.py` | tensor + autograd + optim.SGD |
| 18 | `18_regression_torch_2.py` | nn.Linear + nn.MSELoss + optim |

### Part 3. 신경망 레이어 (19 ~ 38)

| # | 파일 | 주제 |
|---:|---|---|
| 19~20 | `19_linear_1.py`, `20_linear_2.py` | Linear / nn.Sequential |
| 21~22 | `21_activation_1.py`, `22_activation_2.py` | 활성함수 (ReLU/Sigmoid/Tanh) |
| 23~24 | `23_dropout_1.py`, `24_dropout_2.py` | Dropout / train()/eval() 차이 |
| 25~30 | `25_conv_1.py` ~ `30_conv_6.py` | Conv2d (kernel/channels/padding/stride) |
| 31 | `31_linear_vs_conv.py` | Linear vs Conv 파라미터 수 비교 |
| 32 | `32_receptive_field.py` | Receptive Field |
| 33 | `33_1by1conv.py` | 1x1 Conv |
| 34 | `34_maxpool.py` | MaxPool2d |
| 35 | `35_batchnorm.py` | BatchNorm2d (γ, β / train·eval 차이) |
| 36~37 | `36_all_layer_1.py`, `37_all_layer_2.py` | 전체 레이어 통합 (Sequential / nn.Module) |
| 38 | `38_magic_method.py` | Magic method (nn.Module 의 기반) |

### Part 4. 학습 도구 / 함정 (39 ~ 44)

| # | 파일 | 주제 |
|---:|---|---|
| 39 | `39_loss.py` | 손실 함수 비교 (MSE/L1/BCE/CE) |
| 40 | `40_optimizer.py` | 옵티마이저 비교 (SGD/Momentum/Adam) |
| 41 | `41_dataset_dataloader.py` | Dataset/DataLoader 표준 패턴 |
| 42 | `42_save_load.py` | state_dict 저장/불러오기 / 체크포인트 |
| 43 | `43_autograd.py` | requires_grad / backward / no_grad / detach |
| 44 | `44_softmax_crossentropy.py` | ⚠️ Softmax + CrossEntropyLoss 이중 적용 함정 |

## 함수 매핑 요약표 (Part 1 통합)

| 작업 | NumPy | PyTorch |
|---|---|---|
| Tensor 생성 | `np.array([...])` | `torch.tensor([...])` |
| 정규분포 난수 | `np.random.randn(*shape)` | `torch.randn(*shape)` |
| 정수 난수 | `np.random.randint(low, high, shape)` | `torch.randint(low, high, shape)` |
| 차원 수 | `x.ndim` | `t.dim()` |
| 차원 크기 | `x.shape` | `t.shape` (또는 `t.size()`) |
| dtype 확인 | `x.dtype` | `t.dtype` |
| dtype 변경 | `x.astype(np.int32)` | `t.to(torch.int32)` |
| Element-wise | `a + b` 등 | `a + b` 등 |
| 벡터 내적 (1D) | `np.dot(a, b)` | `torch.dot(a, b)` *(1D 전용)* |
| 행렬 곱 | `np.matmul(A, B)` / `A @ B` | `torch.matmul(A, B)` / `A @ B` |
| 배치 행렬 곱 | `np.matmul` (앞 축이 배치) | `torch.matmul` / `torch.bmm` |
| 외적 (1D→2D) | `np.outer(a, b)` | `torch.outer(a, b)` |
| Einstein 합 | `np.einsum('ij,jk->ik', A, B)` | `torch.einsum('ij,jk->ik', A, B)` |
| reshape | `np.reshape(x, s)` / `x.reshape(s)` | `torch.reshape(t, s)` / `t.reshape(s)` / `t.view(s)` |
| 차원 추가 | `np.expand_dims(x, axis=k)` | `torch.unsqueeze(t, k)` / `t.unsqueeze(k)` |
| 차원 제거 | `np.squeeze(x, axis=k)` | `torch.squeeze(t, k)` / `t.squeeze(k)` |
| 차원 순서 변경 | `np.transpose(x, axes)` | `torch.permute(t, dims)` / `t.permute(*dims)` |
| 두 차원 교환 | `np.swapaxes(x, a, b)` | `torch.transpose(t, a, b)` |
| 합치기 (기존 차원) | `np.concatenate([x, y], axis=k)` | `torch.cat([x, y], dim=k)` |
| 쌓기 (새 차원) | `np.stack([x, y], axis=k)` | `torch.stack([x, y], dim=k)` |
| 깊은 복사 | `x.copy()` | `t.clone()` |
| 셔플 | `np.random.shuffle(x)` (in-place) | `x[torch.randperm(N)]` (인덱싱) |
| numpy ↔ torch | — | `torch.from_numpy(x)` / `t.detach().cpu().numpy()` |
| device 이동 | (없음 — CPU 전용) | `t.to("cuda")` / `t.cpu()` |

## 핵심 차이점

- **기본 실수 dtype**: NumPy = `float64`, PyTorch = `float32`
  - 변환 시 dtype을 명시적으로 맞추지 않으면 학습 속도/정확도에 영향이 갈 수 있음
- **`*` 는 element-wise**: 행렬 곱은 `@` 또는 `matmul` ([04 참고](04_matrix_ops.py))
- **`np.dot` vs `torch.dot`**: NumPy는 다차원에서 matmul처럼 동작하지만 PyTorch의 `torch.dot`은 1D 전용
- **Broadcasting 함정**: `(N,)` vs `(N, 1)` — 의도치 않은 `(N, N)` outer-sum 발생 가능
- **reshape vs view (PyTorch)**: `view`는 메모리가 연속(`contiguous`)일 때만 가능. 안전하게는 `reshape` 사용
- **메모리 공유**: 양쪽 모두 reshape 결과는 view(공유)일 수 있음 → 의도치 않은 부작용을 막으려면 `copy()` / `clone()` 사용
- **device**: NumPy는 항상 CPU. PyTorch는 모든 tensor에 `device` 속성을 두고 CPU/GPU 사이를 명시적으로 이동
- **autograd**: PyTorch tensor는 연산 그래프를 가질 수 있음. numpy 변환 전 `detach()` 가 필요

## 실습 방법

```bash
# 단일 파일 실행
python tutorial/01_rank.py

# 전체 순차 실행 (zsh)
for f in tutorial/[0-9][0-9]_*.py; do echo "=== $f ==="; python "$f"; done
```
