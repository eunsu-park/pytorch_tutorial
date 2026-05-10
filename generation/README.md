# Generation

Solar EUV Image → Solar Magnetogram 변환 모델 (Pix2Pix / Conditional GAN).

## Tree

```bash
generation
├── options.py     # argparse 기반 옵션 (BaseOptions/TrainOptions/TestOptions)
├── pipeline.py    # Dataset/DataLoader 정의
├── networks.py    # PatchDiscriminator, PixelDiscriminator, U-Net Generator
├── train.py       # 학습 루프 (Discriminator + Generator 교대 학습)
├── test.py        # 평가 루프 (L1 거리 + inp/tar/gen 비교 영상 저장)
└── utils.py       # fix_seed, get_num_params
```

## 데이터셋 가정

- 폭이 2048 픽셀인 한 이미지에 좌(EUV 입력)·우(Magnetogram 정답) 가 붙어 있는 형식
- `pipeline.py` 의 `LoadData` 가 좌/우 1024 픽셀씩 자름
- 디렉토리 구성:
  ```
  data_root/
  ├── Train/*.png
  └── Test/*.png
  ```

## 학습

```bash
python train.py --name my_run --num_epochs 10 --data_root /path/to/data --save_root ./results
```

## 평가

```bash
python test.py --name my_run --epoch_test 5 --data_root /path/to/data --save_root ./results
```

평가 결과는 `./results/<name>/test_result/sample_NNNN.png` 로 저장됩니다.
한 장에 (입력, 정답, 생성) 이 가로로 이어붙여 비교가 쉽도록 되어 있습니다.

## 핵심 구현 메모

- **Conditional GAN**: Discriminator 는 `(입력, 출력)` 쌍을 받아 진짜/가짜를 판별
  → `train.py` 에서 `torch.cat([inp, tar], dim=1)`, `torch.cat([inp, gen], dim=1)` 사용
- **BCEWithLogitsLoss + use_sigmoid_D=False**: sigmoid 와 BCE 를 한 번에 계산해 수치 안정
- **L1 보조 손실**: `loss_G + lamb * loss_L1` (lamb 기본값 10) — Pix2Pix 표준 패턴
