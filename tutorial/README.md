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
| 04 | `04_numpy_vs_pytorch.py` | array vs tensor 생성과 dtype |
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

### Part 2. PyTorch 회귀 / 신경망 구성요소 (16 ~ 38)

회귀 학습 흐름(16 ~ 18) → Linear / 활성함수 / Dropout / Convolution / Pooling / BatchNorm 등의 레이어 학습 → 전체 레이어 통합 / magic method.

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
