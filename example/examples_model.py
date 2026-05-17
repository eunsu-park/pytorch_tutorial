# examples_model.py
# 실전 패턴 — 물리 정보(forward model)를 내장한 커스텀 nn.Module
#
# 학습 목표
#   examples_dataset.py 가 "데이터 → tensor" 흐름이었다면,
#   여기서는 "관측량 → 물리량(역문제)" 을 푸는 신경망을 직접 만들어 본다.
#
#   소재 : 태양 DEM(Differential Emission Measure) 복원
#     - 입력  : 6개 파장(EUV) 관측 세기            shape (B, num_wave)
#     - 출력  : 43개 온도구간의 DEM 분포            shape (B, num_tbin)
#     - 그리고 그 DEM 으로 다시 EUV 를 재구성(recon)해 함께 반환
#       → "예측 DEM 이 원래 관측을 설명하는가?" 를 학습에 쓸 수 있다.
#
#   배우는 것
#     (1) register_buffer : 학습되지 않지만 모델과 함께 이동/저장되는 상수 텐서
#     (2) forward 안에 물리 forward model(행렬곱) 을 그대로 끼워 넣기
#     (3) 출력이 두 개(dem, recon)인 모델의 호출/형태 확인
#
# 필요 패키지 : torch numpy

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════
#  (1) 모델 정의 — DEMNet
# ══════════════════════════════════════════════════════════════════

# 물리 forward model (이산화) :
#     I(λ) = Σ_T  R(λ, T) · DEM(T) · ΔT
#   - I(λ)            : 파장 λ 의 관측 세기            (num_wave,)
#   - R(λ, T)         : 기기 응답함수 response_function (num_wave, num_tbin)
#   - DEM(T)          : 온도 T 의 방출량 (구하려는 값)  (num_tbin,)
#   - ΔT              : 온도구간 폭 delta_temperature   (num_tbin,)
#
# DEMNet 은 그 "역문제" (관측 I → DEM) 를 MLP 로 근사하고,
# 구한 DEM 을 위 식에 다시 넣어 recon(=재구성된 I) 도 함께 돌려준다.
class DEMNet(nn.Module):
    def __init__(self, response_function,
                 delta_temperature,
                 num_wave=6, num_tbin=43, num_hidden=128):
        super().__init__()

        # register_buffer : 가중치는 아니지만(=requires_grad 안 함)
        #   .to(device) / state_dict 저장 시 모델과 함께 따라다니는 상수.
        #   물리 상수(응답함수·온도폭)를 여기 등록해 두면 forward 에서 바로 쓴다.
        self.register_buffer(
            "response_function",
            torch.as_tensor(response_function, dtype=torch.float32)
        )

        self.register_buffer(
            "delta_temperature",
            torch.as_tensor(delta_temperature, dtype=torch.float32)
        )

        # 관측(num_wave) → 은닉(num_hidden) → DEM(num_tbin) 으로 가는 단순 MLP
        model = []
        model += [nn.Linear(num_wave,  num_hidden)]
        model += [nn.ReLU()]
        model += [nn.Linear(num_hidden, num_tbin)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # x : (B, num_wave)  — 관측된 EUV 세기
        dem = self.model(x)                          # (B, num_tbin)  — 예측 DEM

        # 예측 DEM 을 물리 forward model 에 다시 통과 → 관측 재구성(recon)
        #   dem * ΔT                       : (B, num_tbin)  (브로드캐스트)
        #   ... @ response_function.T      : (B, num_tbin) @ (num_tbin, num_wave)
        #                                  → (B, num_wave) — 입력과 같은 형태
        recon = dem * self.delta_temperature
        recon = recon @ self.response_function.T
        return dem, recon


# ══════════════════════════════════════════════════════════════════
#  (2) 동작 확인
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    BATCH_SIZE = 64
    NUM_WAVE = 6
    NUM_TBIN = 43
    NUM_HIDDEN = 128

    # ────────────── (2-1) 물리 상수 준비 ──────────────
    # 실전에서는 기기 캘리브레이션에서 주어진다. 여기서는 데모용 난수.
    print("[1] 물리 상수 (응답함수 · 온도구간 폭)")

    delta_temperature = np.random.normal(0, 1, (NUM_TBIN))
    response_function = np.random.normal(0, 1, (NUM_WAVE, NUM_TBIN))
    print(f"  delta_temperature : {delta_temperature.shape}   (num_tbin,)")
    print(f"  response_function : {response_function.shape}   (num_wave, num_tbin)")
    print("")

    # ────────────── (2-2) forward model 을 numpy 로 손계산 ──────────────
    # 모델 내부의 recon 계산이 무엇인지 먼저 numpy 로 그대로 재현해 본다.
    print("[2] 물리 forward model 손계산 (numpy)")

    my_euv = np.random.normal(0, 1, (BATCH_SIZE, NUM_WAVE))      # 가짜 관측
    my_dem = np.random.normal(0, 1, (BATCH_SIZE, NUM_TBIN))      # 가짜 DEM
    print(f"  my_euv : {my_euv.shape}   (B, num_wave)")
    print(f"  my_dem : {my_dem.shape}   (B, num_tbin)")

    recon = my_dem * delta_temperature                          # (B, num_tbin)
    print(f"  my_dem * ΔT            → {recon.shape}")
    recon = recon @ response_function.T                         # (B, num_wave)
    print(f"  ... @ response_funcᵀ   → {recon.shape}   (입력 num_wave 로 복귀)")
    print("")

    # ────────────── (2-3) DEMNet 인스턴스화 + forward ──────────────
    # 위 손계산이 forward 안에 그대로 들어있다. 모델로 한 번 돌려 형태 확인.
    print("[3] DEMNet 동작 확인")

    model = DEMNet(
        response_function=response_function,
        delta_temperature=delta_temperature,
        num_wave=NUM_WAVE, num_tbin=NUM_TBIN, num_hidden=NUM_HIDDEN,
    )

    x = torch.as_tensor(my_euv, dtype=torch.float32)            # (B, num_wave)
    dem, recon = model(x)
    print(f"  입력  x     : {tuple(x.shape)}     (B, num_wave)")
    print(f"  출력  dem   : {tuple(dem.shape)}    (B, num_tbin)")
    print(f"  출력  recon : {tuple(recon.shape)}     (B, num_wave) ← 입력과 같은 형태")

    # buffer 는 파라미터가 아니므로 named_parameters 에 안 잡힌다 (학습 대상 X).
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  학습 파라미터 수 : {n_param:,}  (응답함수·ΔT 는 buffer 라 제외)")
    print("")

    # ────────────── 정리 ──────────────
    # 물리 정보 내장 모델 패턴 요약
    #   1. register_buffer : 학습 안 하지만 모델과 함께 이동/저장할 상수 텐서
    #                        (캘리브레이션 행렬, 정규화 통계, 마스크 등)
    #   2. forward 안에서 신경망 출력에 물리 forward model(행렬곱)을 이어 붙여
    #      관측을 재구성 → "물리적으로 말이 되는가" 를 손실로 쓸 수 있다
    #   3. 출력이 여러 개면 tuple 로 반환, 호출부는 dem, recon = model(x)
    #
    # 학습 시(다음 단계 예시)
    #   loss = MSE(dem, dem_true)              # 직접 지도(있다면)
    #        + λ * MSE(recon, x)               # 물리 일관성(관측 재구성)
    #   → register_buffer 덕분에 model.to(device) 한 번이면 상수도 같이 GPU 로.
