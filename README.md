# CBNU / 2024 / 인공지능천문학 I

PyTorch 입문부터 실제 천문학 응용 모델까지 단계적으로 학습하는 교육용 저장소.

## 디렉토리 구조

```bash
.
├── tutorial/
│   ├── tensor/              # 텐서 다루기 + autograd + 매직 메서드 (14 챕터)
│   └── nn/                  # 신경망 회귀·레이어·모델·학습 인프라 (14 챕터)
├── classification/          # Solar Magnetogram → Solar Flare 발생 예측
└── generation/              # Solar EUV → Solar Magnetogram 변환 (Pix2Pix)
```

## tutorial

`tensor/` 폴더에서 NumPy ↔ PyTorch 텐서 다루기와 autograd 까지 익히고,
`nn/` 폴더에서 회귀·신경망 레이어·모델 조립·학습 인프라를 단계적으로 학습합니다.
자세한 챕터 목록은 [`tutorial/README.md`](tutorial/README.md) 참고.

## classification

Solar Magnetogram을 이용한 Solar Flare Occurrence 예측 모델 (CNN 분류).
[`classification/README.md`](classification/README.md) 참고.

## generation

Solar EUV Image → Solar Magnetogram Translation 모델 (Pix2Pix / Conditional GAN).
[`generation/README.md`](generation/README.md) 참고.

## 환경

```bash
conda create -n pytorch_tutorial python=3.10
conda activate pytorch_tutorial
pip install torch numpy matplotlib pandas imageio scikit-image
```

권장 버전 : `torch>=2.0`, `numpy>=1.24`. GPU(CUDA) 사용 시 PyTorch 공식 가이드에 맞는 빌드를 설치.

## 실습 권장 순서

1. `tutorial/tensor/01` ~ `12` : 텐서 기본·연산·형태 변환
2. `tutorial/tensor/13` ~ `14` : autograd 와 nn.Module 의 기반 매직 메서드
3. `tutorial/nn/01` ~ `02` : 같은 회귀 문제를 두 단계 추상화로 풀기
4. `tutorial/nn/03` ~ `11` : 신경망 레이어와 모델 조립
5. `tutorial/nn/12` ~ `14` : Dataset/DataLoader, 손실/옵티마이저, 모델 저장
6. `classification/` : 실제 분류 모델 학습/평가
7. `generation/` : Pix2Pix 기반 영상 변환
