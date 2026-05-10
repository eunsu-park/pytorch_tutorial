# 08_evaluation.py
# 모델 평가 — 검증/테스트 루프와 평가 지표
#
# 학습 목표
#   - 평가 시점의 표준 패턴 : model.eval() + torch.no_grad()
#   - 회귀 평가 지표 : MSE / MAE
#   - 분류 평가 지표 : accuracy / confusion matrix / precision / recall / F1

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)


# ────────────── (1) 평가 시 두 가지 핵심 ──────────────
# 1. model.eval()       : Dropout/BatchNorm 등이 평가 모드로 동작 (예측 안정화)
# 2. with torch.no_grad(): 자동 미분 그래프 생성 안 함 → 메모리 절약 + 속도 향상

print("[1] 평가 시 표준 두 줄")
print("""
  model.eval()
  with torch.no_grad():
      for inp, tar in valid_loader:
          out = model(inp)
          ...
""")

# ────────────── (2) 회귀 평가 ──────────────
# MSE = 평균 제곱 오차, MAE = 평균 절대 오차
print("[2] 회귀 평가")

# 학습된 모델이라고 가정하고, 가짜 결과로 시연
y_pred = torch.tensor([2.1, 1.9, 3.2, 4.0, 5.1])
y_true = torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0])

mse = nn.MSELoss()(y_pred, y_true).item()
mae = nn.L1Loss()(y_pred, y_true).item()
rmse = mse ** 0.5
print(f"  MSE  = {mse:.4f}    (제곱 오차의 평균)")
print(f"  RMSE = {rmse:.4f}    (제곱근 — 원래 단위)")
print(f"  MAE  = {mae:.4f}    (절대 오차의 평균 — 이상치에 강건)")
print("")

# ────────────── (3) 분류 평가 — 혼동행렬 ──────────────
# 가짜 분류 결과 (3-class) 로 시연
print("[3] 분류 평가 — 혼동행렬")

# 모델 출력은 logit. 예측 클래스는 argmax.
logits = torch.tensor([
    [2.0, 0.5, -1.0],   # → 0
    [0.1, 1.5,  0.3],   # → 1
    [0.0, 0.0,  3.0],   # → 2
    [1.5, 1.6,  0.0],   # → 1
    [1.0, 0.5,  0.6],   # → 0
    [-1.0, 2.0, 0.0],   # → 1
])
target = torch.tensor([0, 1, 2, 1, 1, 0])
pred   = logits.argmax(dim=1)

print(f"  예측 = {pred.tolist()}")
print(f"  정답 = {target.tolist()}")
print("")

# 혼동행렬 직접 계산 — sklearn 없이도 가능
def confusion_matrix(y_true, y_pred, num_classes):
    """cm[i, j] = 실제 i 인데 j 로 예측한 샘플 수"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


cm = confusion_matrix(target.tolist(), pred.tolist(), num_classes=3)
print("  Confusion Matrix (행=실제, 열=예측)")
print(f"           pred 0  pred 1  pred 2")
for i, row in enumerate(cm):
    print(f"  true {i}  " + "  ".join(f"{v:>6d}" for v in row))
print("")

# ────────────── (4) 분류 평가 — accuracy / precision / recall / F1 ──────────────
print("[4] 분류 평가 — 클래스별 지표")

accuracy = (pred == target).float().mean().item()
print(f"  accuracy = {accuracy:.4f}    (전체 정답률)")
print("")

# 클래스별 precision / recall / F1
def per_class_metrics(cm):
    num_classes = cm.shape[0]
    metrics = []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp     # 다른 클래스를 k 로 잘못 예측
        fn = cm[k, :].sum() - tp     # k 인데 다른 클래스로 잘못 예측
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.append((precision, recall, f1))
    return metrics


metrics = per_class_metrics(cm)
print(f"  {'class':>6s}  {'precision':>10s}  {'recall':>8s}  {'f1':>6s}")
for k, (p, r, f1) in enumerate(metrics):
    print(f"  {k:>6d}  {p:>10.4f}  {r:>8.4f}  {f1:>6.4f}")
print("")

# 평균 — macro (단순 평균) / weighted (샘플 수 가중)
macro_f1 = sum(m[2] for m in metrics) / len(metrics)
support  = cm.sum(axis=1)
weighted_f1 = sum(m[2] * s for m, s in zip(metrics, support)) / support.sum()
print(f"  macro    F1 = {macro_f1:.4f}    (클래스 단순 평균)")
print(f"  weighted F1 = {weighted_f1:.4f}    (샘플 수로 가중 — 불균형 데이터에 적합)")
print("")

# ────────────── (5) 검증 루프 표준 패턴 ──────────────
print("[5] 검증 루프 — 학습 루프 안에 끼워 넣는 패턴")
print("""
  for epoch in range(N):
      # 학습
      model.train()
      for inp, tar in train_loader: ...

      # 검증
      model.eval()
      losses, all_pred, all_target = [], [], []
      with torch.no_grad():
          for inp, tar in valid_loader:
              out = model(inp)
              losses.append(criterion(out, tar).item())
              all_pred.append(out.argmax(dim=1).cpu())
              all_target.append(tar.cpu())
      val_loss = np.mean(losses)
      val_pred = torch.cat(all_pred).numpy()
      val_true = torch.cat(all_target).numpy()
      cm = confusion_matrix(val_true, val_pred, num_classes)
      ...
""")

# ────────────── 정리 ──────────────
# 평가 시 두 가지 필수
#   model.eval()           : Dropout/BN 모드 전환
#   with torch.no_grad():  : 그래프 생성 차단 (메모리·속도 절약)
#
# 회귀 지표
#   MSE  : 제곱 오차 평균
#   RMSE : MSE 의 제곱근 — 원래 단위로 해석 가능
#   MAE  : 절대 오차 평균 — 이상치에 강건
#
# 분류 지표
#   accuracy           : 전체 정답률 — 균형 잡힌 데이터에 적합
#   confusion matrix    : 어떤 클래스를 어떤 클래스로 헷갈리는지 한눈에
#   precision/recall/F1 : 클래스별로 — 불균형 데이터에 필수
#   macro F1           : 클래스 단순 평균
#   weighted F1        : 샘플 수 가중 평균 — 불균형 데이터에 더 의미 있음
#
# 실전 팁
#   - 학습 중 매 epoch 검증 → 가장 좋은 val 지표일 때만 모델 저장 (early stopping)
#   - test_loader 는 학습 종료 후 마지막에 단 한 번만 — 모델 선택에 영향 없게
