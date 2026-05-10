## 내장 라이브러리 import
import os
import time

## 외부 라이브러리 import
import torch
from torch.utils.data import DataLoader, random_split
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

## dataset 정의 후 train / val 로 split
full_dataset, _ = define_dataset(opt)
n_total = len(full_dataset)
n_val = max(1, int(n_total * opt.val_ratio))
n_train = n_total - n_val
train_set, val_set = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(opt.seed),
)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,  num_workers=opt.num_workers)
val_loader   = DataLoader(val_set,   batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
print(f"Train: {len(train_set)} samples / Val: {len(val_set)} samples")

## 학습 결과 저장 디렉토리 생성
save_dir = os.path.join(opt.save_root, opt.name)
os.makedirs(save_dir, exist_ok=True)

## 학습 시작
iters = 0
epochs = 0
running_losses = []
best_val_loss = float("inf")
t0 = time.time()

while epochs < opt.num_epochs:

    network.train()
    for idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        # 실제 학습이 이루어지는 부분
        optimizer.zero_grad()
        output = network(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # 실제 학습이 이루어지는 부분

        running_losses.append(loss.item())
        iters += 1

        if iters % 100 == 0:
            print(f"Epoch [{epochs}/{opt.num_epochs}], Step [{iters}], "
                  f"Train Loss: {np.mean(running_losses):.4f}, Time: {time.time()-t0:.4f}")
            running_losses = []
            t0 = time.time()

    epochs += 1

    ## 매 epoch 종료 시 검증셋으로 평가
    network.eval()
    val_losses = []
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device)
            label = label.to(device)
            output = network(image)
            val_losses.append(criterion(output, label).item())

            pred_class = output.argmax(dim=1)
            true_class = label.argmax(dim=1) if label.ndim > 1 else label
            val_correct += (pred_class == true_class).sum().item()
            val_total   += true_class.size(0)
    val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
    val_acc  = val_correct / max(1, val_total)
    print(f"[Epoch {epochs}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    ## best model 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        state = {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(state, f"{save_dir}/model_best.pt")
        print(f"  ↳ best model 갱신 (val_loss={val_loss:.4f})")

    ## 주기 저장 (5 epoch 단위)
    if epochs % 5 == 0:
        state = {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(state, f"{save_dir}/model_{epochs:04d}.pt")

## 마지막 모델 저장
state = {
    "network": network.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epochs,
}
torch.save(state, f"{save_dir}/model_final.pt")
