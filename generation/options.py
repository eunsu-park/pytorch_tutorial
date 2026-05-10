import argparse
import sys


class BaseOptions():
    """
    BaseOptions 클래스
    Pix2Pix(Conditional GAN) 학습/평가 전반에 걸쳐 사용되는 공통 옵션을 정의한다.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--name', type=str, default='default', help="실험 이름 (저장 디렉토리 이름이 됨)")
        self.parser.add_argument('--seed', type=int, default=1111, help="난수 시드")

        # 입출력 영상 차원
        self.parser.add_argument('--in_channels',  type=int, default=1, help="입력 영상의 채널 수")
        self.parser.add_argument('--out_channels', type=int, default=1, help="출력(목표) 영상의 채널 수")
        self.parser.add_argument('--image_size',   type=int, default=256, help="입력 영상 크기 (정사각형)")

        # Discriminator
        self.parser.add_argument('--nb_layers_D',    type=int, default=3,    help="Discriminator 의 다운샘플링 블록 수 (0 이면 PixelDiscriminator)")
        self.parser.add_argument('--nb_feat_init_D', type=int, default=64,   help="Discriminator 첫 블록 feature 수")
        self.parser.add_argument('--nb_feat_max_D',  type=int, default=512,  help="Discriminator feature 수 상한")
        self.parser.add_argument('--use_sigmoid_D',  action='store_true',    help="True 면 Discriminator 출력에 sigmoid (BCELoss 와 짝)")

        # Generator (U-Net)
        self.parser.add_argument('--nb_down_G',      type=int, default=8,    help="U-Net 의 다운샘플링 단계 수")
        self.parser.add_argument('--nb_feat_init_G', type=int, default=64,   help="U-Net 첫 블록 feature 수")
        self.parser.add_argument('--nb_feat_max_G',  type=int, default=512,  help="U-Net feature 수 상한")
        self.parser.add_argument('--no_dropout_G',   dest='use_dropout_G', action='store_false', default=True, help="Generator 의 Dropout 사용 끄기")
        self.parser.add_argument('--use_tanh_G',     action='store_true',    help="Generator 마지막에 Tanh 사용 (출력을 [-1,1] 로)")

        # 초기화 / 옵티마이저
        self.parser.add_argument('--init_type', type=str,   default='normal', help="가중치 초기화 방법 (normal/xavier/kaiming/orthogonal)")
        self.parser.add_argument('--init_gain', type=float, default=0.02,     help="초기화 gain")
        self.parser.add_argument('--lr',        type=float, default=0.0002,   help="학습률")
        self.parser.add_argument('--beta1',     type=float, default=0.5,      help="Adam beta1")
        self.parser.add_argument('--beta2',     type=float, default=0.999,    help="Adam beta2")

        # Pix2Pix L1 가중치
        self.parser.add_argument('--lamb', type=float, default=10.0, help="L1 손실 가중치 (loss = loss_G + lamb * loss_L1)")

        # 경로 (사용자 환경에 맞춰 --data_root, --save_root 로 덮어쓸 것)
        self.parser.add_argument("--data_root", type=str, default="./data",    help="데이터셋 루트 경로 (Train/Test 하위 디렉토리)")
        self.parser.add_argument("--save_root", type=str, default="./results", help="학습 결과·체크포인트 저장 루트")

    def parse(self):
        """일반 스크립트 / Jupyter 모두에서 안전하게 동작하도록 파싱"""
        is_notebook = "ipykernel" in sys.modules or "google.colab" in sys.modules
        if is_notebook:
            opt, _ = self.parser.parse_known_args(args=[])
        else:
            opt, _ = self.parser.parse_known_args()
        return opt


class TrainOptions(BaseOptions):
    """학습에 필요한 옵션"""
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser.add_argument('--is_train',    dest='is_train', action='store_true', default=True, help="학습 모드 (기본값)")
        self.parser.add_argument('--gpu_id',      type=int, default=0, help="사용할 GPU 인덱스")
        self.parser.add_argument('--batch_size',  type=int, default=1, help="배치 크기")
        self.parser.add_argument('--num_workers', type=int, default=2, help="DataLoader 의 worker 수")
        self.parser.add_argument('--num_epochs',  type=int, default=10, help="학습 epoch 수")


class TestOptions(BaseOptions):
    """평가에 필요한 옵션"""
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--is_train',    dest='is_train', action='store_false', default=False, help="평가 모드 (기본값)")
        self.parser.add_argument('--gpu_id',      type=int, default=0, help="사용할 GPU 인덱스")
        self.parser.add_argument('--batch_size',  type=int, default=1, help="배치 크기")
        self.parser.add_argument('--num_workers', type=int, default=2, help="DataLoader 의 worker 수")
        self.parser.add_argument('--epoch_test',  type=int, default=5, help="평가에 사용할 체크포인트 epoch")
