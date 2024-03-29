## 내장 라이브러리 import
import os
import time

## 외부 라이브러리 import
import torch
import numpy as np

## 제작한 모듈 import
from options import TrainOptions
from networks import define_network, define_criterion, define_optimizer
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

## network, criterion, optimizer 정의
network = define_network(opt).to(device)
criterion = define_criterion(opt).to(device)
optimizer = define_optimizer(network, opt)
print(network)
print(f"Number of parameters : {get_num_params(network)}")
print(criterion)
print(optimizer)

## dataset, dataloader 정의
dataset, dataloader = define_dataset(opt)
print(len(dataset), len(dataloader))

## 학습 결과 저장 디렉토리 생성
save_dir = os.path.join(opt.save_root, opt.name)
os.makedirs(save_dir, exist_ok=True)

## 학습 시작
network.train()
iters = 0
epochs = 0
losses = []
t0 = time.time()

while epochs < opt.num_epochs:

    for idx, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)

        # 실제 학습이 이루어지는 부분
        optimizer.zero_grad()
        output = network(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # 실제 학습이 이루어지는 부분

        losses.append(loss.item())            
        iters += 1

        if iters % 100 == 0:
            print(f"Epoch [{epochs}/{opt.num_epochs}], Step [{iters}], Loss: {np.mean(losses):.4f}, Time: {time.time()-t0:.4f}")
            losses = []
            t0 = time.time()

    epochs += 1

    if epochs % 5 == 0 :
        state_network = network.state_dict()
        state_optimizer = optimizer.state_dict()
        state = {"network": state_network, "optimizer": state_optimizer, "epoch": epochs}
        torch.save(state, f"{save_dir}/model_{epochs:04d}.pt")

state_network = network.state_dict()
state_optimizer = optimizer.state_dict()
state = {"network": state_network, "optimizer": state_optimizer, "epoch": epochs}
torch.save(state, f"{save_dir}/model_final.pt")
