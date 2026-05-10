# CBNU / 2024 / 인공지능천문학 I

PyTorch 입문부터 실제 천문학 응용 모델까지 단계적으로 학습하는 교육용 저장소.

## 디렉토리 구조

```bash
.
├── tutorial/                # PyTorch 기초 (28 챕터)
├── classification/          # Solar Magnetogram → Solar Flare 발생 예측
└── generation/              # Solar EUV → Solar Magnetogram 변환 (Pix2Pix)
```

## tutorial

NumPy ↔ PyTorch 텐서 기본/연산/형태 변환(01~12), autograd·magic_method(13~14),
회귀 학습 흐름(15~16), 신경망 레이어(17~22), 모델 조립(23~25), 학습 인프라(26~28).
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

1. `tutorial/01` ~ `12` : 텐서 기본·연산·형태 변환
2. `tutorial/13` ~ `14` : autograd 와 nn.Module 의 기반 매직 메서드
3. `tutorial/15` ~ `16` : 같은 회귀 문제를 두 단계 추상화로 풀기
4. `tutorial/17` ~ `25` : 신경망 레이어와 모델 조립
5. `tutorial/26` ~ `28` : Dataset/DataLoader, 손실/옵티마이저, 모델 저장
6. `classification/` : 실제 분류 모델 학습/평가
7. `generation/` : Pix2Pix 기반 영상 변환
