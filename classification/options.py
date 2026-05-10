import argparse
import sys


class BaseOptions():
    """
    BaseOptions 클래스
    모델 개발 전반에 걸쳐 사용되는 옵션을 정의하는 클래스.
    공통 옵션(경로, 시드, 입력 차원 등)을 여기서 정의하고
    TrainOptions / TestOptions 가 각자 학습·평가에 필요한 옵션을 추가한다.
    """
    def __init__(self):
        """
        argparse.ArgumentParser 객체를 생성하고, 기본 옵션을 등록한다.
        """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--name', type=str, default='default', help="실험 이름 (저장 디렉토리 이름이 됨)")
        self.parser.add_argument('--seed', type=int, default=1111, help="난수 시드")
        self.parser.add_argument('--in_channels', type=int, default=1, help="입력 영상의 채널 수")
        self.parser.add_argument('--num_classes', type=int, default=2, help="분류 클래스 수")
        self.parser.add_argument('--image_size', type=int, default=224, help="입력 영상 크기 (정사각형)")

        # 데이터/결과 경로는 상대경로 default. 사용자 환경에 맞게 --data_root, --save_root 로 덮어쓸 것.
        self.parser.add_argument("--data_root", type=str, default="./data",   help="데이터셋 루트 경로 (Train.csv, Test.csv 가 있는 위치)")
        self.parser.add_argument("--save_root", type=str, default="./results", help="학습 결과·체크포인트 저장 루트")


    def parse(self):
        """
        입력된 인자를 파싱한다.
        - 일반 스크립트 실행 시   : sys.argv 의 인자를 파싱
        - Jupyter / IPython 환경 : argv 가 비정상이라 parse_args() 가 실패할 수 있으므로
                                    parse_known_args 로 안전하게 처리하고 알려지지 않은 인자는 무시
        Returns:
            argparse.Namespace
        """
        # IPython/Jupyter 에서는 sys.argv 에 알 수 없는 인자가 끼어 있을 수 있어 known_args 만 사용
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
        # bool 옵션은 type=bool 사용 시 'False' 문자열도 True 로 평가되는 동작을 피하기 위해
        # store_true / store_false 패턴 사용
        self.parser.add_argument('--is_train', dest='is_train', action='store_true', default=True, help="학습 모드 (기본값)")
        self.parser.add_argument('--gpu_id', type=int, default=0, help="사용할 GPU 인덱스")
        self.parser.add_argument('--batch_size', type=int, default=4, help="배치 크기")
        self.parser.add_argument('--num_workers', type=int, default=2, help="DataLoader 의 worker 수")
        self.parser.add_argument('--num_epochs', type=int, default=10, help="학습 epoch 수")
        self.parser.add_argument('--lr', type=float, default=0.0002, help="학습률")
        self.parser.add_argument('--val_ratio', type=float, default=0.1, help="학습셋에서 떼어낼 검증셋 비율")


class TestOptions(BaseOptions):
    """평가에 필요한 옵션"""
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--is_train', dest='is_train', action='store_false', default=False, help="평가 모드 (기본값)")
        self.parser.add_argument('--gpu_id', type=int, default=0, help="사용할 GPU 인덱스")
        self.parser.add_argument('--batch_size', type=int, default=1, help="배치 크기")
        self.parser.add_argument('--num_workers', type=int, default=0, help="DataLoader 의 worker 수")
        self.parser.add_argument('--epoch_test', type=int, default=5, help="평가에 사용할 체크포인트 epoch")
