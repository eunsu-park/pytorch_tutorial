## 내장 라이브러리 import
import os

## 외부 라이브러리 import
import torch
import numpy as np
from imageio import imwrite

## 제작한 모듈 import
from options import TestOptions
from networks import define_generator
from pipeline import define_dataset

## options 설정
opt = TestOptions().parse()

## device 설정
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

## 저장된 모델 불러오기
save_dir = os.path.join(opt.save_root, opt.name)
save_dir_model = os.path.join(save_dir, "model")
save_dir_result = os.path.join(save_dir, "test_result")
os.makedirs(save_dir_result, exist_ok=True)

ckpt_path = f"{save_dir_model}/model_{opt.epoch_test:04d}.pt"
state = torch.load(ckpt_path, map_location=device)
print(f"Loaded checkpoint : {ckpt_path}")

## generator 만 사용 (test 시에는 discriminator 불필요)
generator = define_generator(opt, state_dict=state["generator"]).to(device)
generator.eval()

## dataset, dataloader 정의 (TestOptions 의 is_train=False → Test 셋)
dataset, dataloader = define_dataset(opt)
print(f"Number of test samples : {len(dataset)}")

## L1 metric 누적용
l1_losses = []

with torch.no_grad():
    for idx, (inp, tar) in enumerate(dataloader):
        inp = inp.to(device)
        tar = tar.to(device)

        gen = generator(inp)

        ## L1 거리(픽셀 차이의 절댓값 평균) — 생성 품질의 간이 지표
        l1 = torch.mean(torch.abs(gen - tar)).item()
        l1_losses.append(l1)

        ## [-1, 1] 정규화된 출력을 [0, 255] uint8 영상으로 복원하여 저장
        # inp / tar / gen 을 가로로 이어붙여 한 장으로 비교
        def to_uint8_image(t):
            t = t.detach().cpu().numpy()[0]            # (C, H, W)
            t = (t + 1.0) * 127.5                       # [-1, 1] → [0, 255]
            t = np.clip(t, 0, 255).astype(np.uint8)
            if t.shape[0] == 1:                         # 단일 채널은 (H, W) 로 squeeze
                t = t[0]
            else:
                t = np.transpose(t, (1, 2, 0))          # (H, W, C)
            return t

        triplet = np.concatenate(
            [to_uint8_image(inp), to_uint8_image(tar), to_uint8_image(gen)],
            axis=1,
        )
        imwrite(f"{save_dir_result}/sample_{idx:04d}.png", triplet)

print(f"Average L1 distance : {np.mean(l1_losses):.4f}")
print(f"결과 저장 위치 : {save_dir_result}")
