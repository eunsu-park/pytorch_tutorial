## 내장 라이브러리 import
import os

## 외부 라이브러리 import
import torch
import numpy as np

## 제작한 모듈 import
from options import TestOptions
from networks import define_network, define_criterion
from pipeline import define_dataset

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

save_dir = os.path.join(opt.save_root, opt.name)
state = torch.load(f"{save_dir}/model_{opt.epoch_test:04d}.pt", map_location=device)
print(f"Loaded checkpoint : epoch={state.get('epoch')}, "
      f"val_loss={state.get('val_loss')}, val_acc={state.get('val_acc')}")

network = define_network(opt, state_dict=state["network"]).to(device)
criterion = define_criterion(opt).to(device)

dataset, dataloader = define_dataset(opt)
print(f"Number of test samples : {len(dataset)}")

losses = []
labels = []
predictions = []

network.eval()
with torch.no_grad():
    for idx, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        output = network(image)
        losses.append(criterion(output, label).item())

        labels.append(label.detach().cpu().numpy())
        predictions.append(output.detach().cpu().numpy())

labels = np.concatenate(labels, 0)
predictions = np.concatenate(predictions, 0)

labels_class = np.argmax(labels, 1)
predictions_class = np.argmax(predictions, 1)

# ────────────── 평가 지표 ──────────────
def confusion_matrix(y_true, y_pred, num_classes):
    """
    혼동행렬 계산. shape = (num_classes, num_classes)
    cm[i, j] = 실제 i 인데 j 로 예측한 샘플 수
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm):
    """
    혼동행렬로부터 클래스별 precision / recall / F1 을 계산.
    분모가 0 인 경우는 0 으로 처리 (학습 초기에 한쪽 클래스가 전혀 안 나올 수 있음).
    """
    num_classes = cm.shape[0]
    metrics = []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.append((precision, recall, f1))
    return metrics


num_classes = opt.num_classes
cm = confusion_matrix(labels_class, predictions_class, num_classes)
metrics = precision_recall_f1(cm)

accuracy = (labels_class == predictions_class).mean()

print("")
print(f"Average Loss : {np.mean(losses):.4f}")
print(f"Accuracy     : {accuracy:.4f}")
print("")
print("[Confusion Matrix]    행 = 실제, 열 = 예측")
header = "        " + "  ".join([f"pred{i:>2d}" for i in range(num_classes)])
print(header)
for i, row in enumerate(cm):
    print(f"true{i:>2d}  " + "  ".join([f"{v:>6d}" for v in row]))

print("")
print("[Per-class metrics]")
print(f"  {'class':>6s}  {'precision':>10s}  {'recall':>8s}  {'f1':>6s}")
for k, (p, r, f1) in enumerate(metrics):
    print(f"  {k:>6d}  {p:>10.4f}  {r:>8.4f}  {f1:>6.4f}")
