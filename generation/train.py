## 내장 라이브러리 import
import os
import time

## 외부 라이브러리 import
import torch
import numpy as np

## 제작한 모듈 import
from options import TrainOptions
from networks import define_discriminator, define_generator
from pipeline import define_dataset
from utils import fix_seed, get_num_params

## options 설정
opt = TrainOptions().parse()

## seed 고정
fix_seed(opt.seed)

## device 설정
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

## discriminator, generator 정의
discriminator = define_discriminator(opt).to(device)
generator = define_generator(opt).to(device)
print(discriminator)
print(generator)

## optimizer 정의
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
print(optimizer_D, optimizer_G)

## criterion 정의
# BCEWithLogitsLoss : sigmoid + BCE 를 한 번에 계산하므로 수치적으로 안정적.
# Discriminator 의 use_sigmoid_D=False 와 짝을 이룸.
# (참고: use_sigmoid_D=True 로 두고 nn.BCELoss 를 쓰면 동일 결과지만 수치 안정성이 떨어짐)
criterion = torch.nn.BCEWithLogitsLoss().to(device)
print(criterion)
l1_criterion = torch.nn.L1Loss().to(device)
print(l1_criterion)

## dataset, dataloader 정의
dataset, dataloader = define_dataset(opt)
print(len(dataset), len(dataloader))

## 학습 결과 저장 디렉토리 생성
save_dir = os.path.join(opt.save_root, opt.name)
save_dir_model = os.path.join(save_dir, "model")
save_dir_image = os.path.join(save_dir, "image")
os.makedirs(save_dir_model, exist_ok=True)
os.makedirs(save_dir_image, exist_ok=True)

## 학습 시작
discriminator.train()
generator.train()
iters = 0
epochs = 0
losses_D = []
losses_G = []
losses_L = []
t0 = time.time()

while epochs < opt.num_epochs:

    for idx, (inp, tar) in enumerate(dataloader):
        inp = inp.to(device)
        tar = tar.to(device)
        gen = generator(inp)

        ## Pix2Pix : Discriminator 는 (입력, 출력) 쌍을 받아 판별 (Conditional GAN)
        pair_real = torch.cat([inp, tar], dim=1)            # 진짜 쌍 : (inp, 정답 영상)
        pair_fake = torch.cat([inp, gen.detach()], dim=1)   # 가짜 쌍 : (inp, 생성 영상). detach 로 G 그래프 차단

        ## Discriminator 학습
        optimizer_D.zero_grad()
        pred_real = discriminator(pair_real)
        pred_fake = discriminator(pair_fake)
        loss_D_real = criterion(pred_real, torch.ones_like(pred_real))
        loss_D_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        losses_D.append(loss_D.item())

        ## Generator 학습 — Discriminator 를 속이고(L_G), 정답에 가깝게(L_L1)
        optimizer_G.zero_grad()
        pair_fake_for_G = torch.cat([inp, gen], dim=1)      # 여기서는 detach 하지 않음
        pred_fake = discriminator(pair_fake_for_G)
        loss_G = criterion(pred_fake, torch.ones_like(pred_fake))
        loss_L = l1_criterion(gen, tar)
        loss = loss_G + opt.lamb * loss_L
        loss.backward()
        optimizer_G.step()
        losses_G.append(loss_G.item())
        losses_L.append(loss_L.item())

        iters += 1

        if iters % 100 == 0:
            print(f"Epoch [{epochs}/{opt.num_epochs}], Step [{iters}], "
                  f"Loss_D: {np.mean(losses_D):.4f}, Loss_G: {np.mean(losses_G):.4f}, "
                  f"Loss_L: {np.mean(losses_L):.4f}, Time: {time.time()-t0:.4f}")
            losses_D = []
            losses_G = []
            losses_L = []
            t0 = time.time()

    epochs += 1

    ## epoch 종료 시 모델 + optimizer 상태 저장 (5 epoch 단위)
    if epochs % 5 == 0:
        state = {
            "discriminator": discriminator.state_dict(),
            "generator": generator.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "epoch": epochs,
        }
        torch.save(state, f"{save_dir_model}/model_{epochs:04d}.pt")

## 마지막 모델 저장
state = {
    "discriminator": discriminator.state_dict(),
    "generator": generator.state_dict(),
    "optimizer_D": optimizer_D.state_dict(),
    "optimizer_G": optimizer_G.state_dict(),
    "epoch": epochs,
}
torch.save(state, f"{save_dir_model}/model_final.pt")
