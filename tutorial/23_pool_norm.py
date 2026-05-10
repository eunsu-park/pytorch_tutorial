# 23_pool_norm.py
# Pooling (MaxPool2d) + Normalization (BatchNorm2d)
#
# 두 레이어 모두 학습 가능 파라미터가 거의 없거나(Pool=0) 매우 적지만(BN=2*C),
# 모델의 안정성·일반화에 큰 영향을 미친다.

import torch
import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# ────────────── (1) MaxPool2d — 다운샘플링 ──────────────
# 영역 안에서 최댓값만 남김. 특징을 보존하면서 H, W 를 줄임.
# 학습 가능 파라미터 0.
print("[1] MaxPool2d")

inp = torch.randn(128, 1, 16, 16)

# 표준 다운샘플링 : kernel=2, stride=2 → 정확히 절반
pool = nn.MaxPool2d(kernel_size=2, stride=2)
out = pool(inp)
print(f"  kernel=2, stride=2 : {tuple(inp.shape)} → {tuple(out.shape)}    ← 표준 (절반)")
print(f"  파라미터 수 : {get_num_params(pool)}")

# kernel=2 만 지정하면 stride 도 2 가 default
pool = nn.MaxPool2d(kernel_size=2)
print(f"  kernel=2  (stride 생략) : {tuple(pool(inp).shape)}    ← stride=kernel 이 기본")

# kernel=3, stride=2, padding=1
pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"  kernel=3, stride=2, padding=1 : {tuple(pool(inp).shape)}")
print("")

# 비슷한 것들 :
#   nn.AvgPool2d           — 평균 풀링
#   nn.AdaptiveAvgPool2d   — 출력 크기를 고정 (H_out, W_out 만 지정)
print("  AdaptiveAvgPool2d : 출력 크기를 고정해 입력에 무관하게 만들기")
adapt = nn.AdaptiveAvgPool2d((1, 1))
print(f"  AdaptiveAvgPool2d((1,1)) : {tuple(adapt(inp).shape)}    ← 분류 모델 끝부분에서 자주 사용")
print("")

# ────────────── (2) BatchNorm2d — 채널별 정규화 ──────────────
# 핵심
#   1. 정규화      : 같은 채널 안에서 (B × H × W) 통계로 평균 0, 분산 1
#   2. 학습 파라미터 γ, β (채널마다 1쌍) → 학습 가능 파라미터 수 = 2 * C
#   3. 추적 통계   : running_mean, running_var (학습이 아닌 이동평균)
#   4. train()/eval() 동작 차이
#       train() : 현재 배치의 통계로 정규화 + running 통계 갱신
#       eval()  : 저장된 running 통계로 정규화 (배치 1 이어도 안정)
print("[2] BatchNorm2d")

inp = torch.randn(128, 3, 16, 16)
bn = nn.BatchNorm2d(3)
out = bn(inp)
print(f"  BatchNorm2d(3)  : 출력 {tuple(out.shape)},  파라미터 수 = {get_num_params(bn)}  (= 2 * 3, γ + β)")

inp = torch.randn(128, 64, 16, 16)
bn = nn.BatchNorm2d(64)
print(f"  BatchNorm2d(64) : 파라미터 수 = {get_num_params(bn)} (= 2 * 64)")
print("")

# ────────────── (3) train()/eval() 시연 ──────────────
print("[3] train/eval 동작 차이")

torch.manual_seed(0)
bn = nn.BatchNorm2d(3)

bn.train()
print(f"  [train mode] running 통계 갱신 여부 :")
print(f"    forward 전 running_mean = {bn.running_mean.tolist()}")
_ = bn(torch.randn(8, 3, 16, 16))
print(f"    forward 후 running_mean = {[round(v, 4) for v in bn.running_mean.tolist()]}    ← 갱신됨")

bn.eval()
mean_before = bn.running_mean.clone()
_ = bn(torch.randn(8, 3, 16, 16))
mean_after = bn.running_mean.clone()
print(f"  [eval  mode] forward 후 running_mean 변경됨? {not torch.equal(mean_before, mean_after)}    ← 갱신 안 됨")
print("")

# ────────────── 비교 정리 ──────────────
# Pool 레이어
#   nn.MaxPool2d / nn.AvgPool2d : 영역 단위로 다운샘플링. 학습 파라미터 0.
#   nn.AdaptiveAvgPool2d        : 출력 크기 고정 — 분류 모델 마지막 부근에서 자주 사용.
#
# BatchNorm
#   학습 파라미터 = 2 * num_features  (γ scale, β shift)
#   추적 통계 (running_mean, running_var) 는 학습이 아닌 이동평균으로 갱신
#   train() / eval() 모드 전환 필수 — eval() 을 잊으면 작은 배치에서 결과가 매번 흔들림
#
# 표준 블록 패턴 :  Conv → BN → 활성 → (MaxPool)
