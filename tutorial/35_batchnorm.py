# 35_batchnorm.py
# BatchNorm2d — 채널별 배치 정규화
#
# 핵심
#   1. 정규화      : 같은 채널 안에서 (배치 × H × W) 통계를 사용해 평균 0, 분산 1 로 만든다.
#   2. 학습 파라미터 γ, β (채널마다 1쌍) → 학습 가능한 파라미터 수 = 2 * num_features
#   3. 추적 통계   : running_mean, running_var (학습이 아니라 이동평균으로 갱신, requires_grad=False)
#   4. train()/eval() 동작 차이
#        - train() : 현재 배치의 통계로 정규화 + running 통계 갱신
#        - eval()  : 저장된 running 통계로 정규화 (배치가 1이어도 안정적)

import torch
import torch.nn as nn

def get_num_params(model):
    """모델(레이어)의 학습 가능 파라미터 수를 계산하는 함수"""
    return sum([p.numel() for p in model.parameters()])

# 채널 수 3
inp = torch.randn(128, 3, 16, 16)
layer = nn.BatchNorm2d(3)
out = layer(inp)
print(out.size())
print(f"Number of parameters: {get_num_params(layer)}  ( = 2 * 3, γ + β )")
print("")

# 채널 수 64
inp = torch.randn(128, 64, 16, 16)
layer = nn.BatchNorm2d(64)
out = layer(inp)
print(out.size())
print(f"Number of parameters: {get_num_params(layer)}  ( = 2 * 64, γ + β )")
print("")

# train() vs eval() 차이 시연
torch.manual_seed(0)
layer = nn.BatchNorm2d(3)

layer.train()
print("[train mode] running 통계가 갱신됨")
print("  before forward — running_mean:", layer.running_mean.tolist())
_ = layer(torch.randn(8, 3, 16, 16))
print("  after  forward — running_mean:", [round(x, 4) for x in layer.running_mean.tolist()])

layer.eval()
print("[eval mode]  running 통계로 정규화 (갱신되지 않음)")
mean_before = layer.running_mean.clone()
_ = layer(torch.randn(8, 3, 16, 16))
mean_after = layer.running_mean.clone()
print(f"  변경 여부 : {not torch.equal(mean_before, mean_after)}  (False 면 갱신되지 않은 것)")
