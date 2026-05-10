# CBNU / 2024 / 인공지능천문학 I

PyTorch 입문부터 실제 천문학 응용 모델까지 단계적으로 학습하는 교육용 저장소.

## 디렉토리 구조

```bash
.
├── tutorial/                # PyTorch 기초 (1~44 챕터)
├── classification/          # Solar Magnetogram → Solar Flare 발생 예측
└── generation/              # Solar EUV → Solar Magnetogram 변환 (Pix2Pix)
```

## tutorial

NumPy ↔ PyTorch 비교(01~15), 회귀 학습 흐름(16~18), 신경망 레이어(19~38), 학습 도구·함정(39~44).
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

1. `tutorial/01` ~ `15` : NumPy ↔ PyTorch 차이 익히기
2. `tutorial/16` ~ `18` : 같은 회귀 문제를 세 단계 추상화로 풀기
3. `tutorial/19` ~ `38` : 신경망 구성요소
4. `tutorial/39` ~ `44` : 학습 도구와 흔한 함정
5. `classification/` : 실제 분류 모델 학습/평가
6. `generation/` : Pix2Pix 기반 영상 변환

## 함정 학습 (★ 중요)

`tutorial/44_softmax_crossentropy.py` 는 분류 모델에서 가장 흔한 실수인
`nn.Softmax` + `nn.CrossEntropyLoss` 이중 적용을 보여줍니다.
`classification/networks.py` 에 같은 함정이 의도적으로 보존되어 있으니
44 챕터 학습 후 직접 찾아보세요.
