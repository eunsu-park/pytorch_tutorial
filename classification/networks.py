import torch
import torch.nn as nn
import torch.nn.init as init


class CustomNetwork(nn.Module):
    """
    CustomNetwork 클래스
    """
    def __init__(self, in_channels, num_classes):
        """
        CustomModel 클래스의 생성자
        Args:
            in_channels : int
                입력 이미지의 채널 수
            num_classes : int
                클래스 수
        """
        super(CustomNetwork, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.define_classifier()
        self.define_feature_extractor()

    def define_feature_extractor(self):
        """
        feature extractor를 정의하는 함수
        """
        model = []

        model += [nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
                  nn.BatchNorm2d(64), nn.ReLU()]
        model += [nn.Conv2d(64, 64, kernel_size=3, padding=1),
                  nn.BatchNorm2d(64), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        model += [nn.Conv2d(64, 128, kernel_size=3, padding=1),
                  nn.BatchNorm2d(128), nn.ReLU()]
        model += [nn.Conv2d(128, 128, kernel_size=3, padding=1),
                  nn.BatchNorm2d(128), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        model += [nn.Conv2d(128, 256, kernel_size=3, padding=1),
                  nn.BatchNorm2d(256), nn.ReLU()]
        model += [nn.Conv2d(256, 256, kernel_size=3, padding=1),
                  nn.BatchNorm2d(256), nn.ReLU()]
        model += [nn.Conv2d(256, 256, kernel_size=3, padding=1),
                  nn.BatchNorm2d(256), nn.ReLU()]
        model += [nn.Conv2d(256, 256, kernel_size=3, padding=1),
                  nn.BatchNorm2d(256), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        model += [nn.Conv2d(256, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                  nn.BatchNorm2d(512), nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        self.feature_extractor = nn.Sequential(*model)

    def define_classifier(self):
        """
        classifier를 정의하는 함수

        ⚠️  학습 함정 (의도적으로 보존됨, 학습 후 직접 발견하도록)
        ───────────────────────────────────────────────────────────
        아래 마지막 줄의 nn.Softmax(dim=1) 은 train.py 의 nn.CrossEntropyLoss()
        와 함께 쓰면 'softmax 가 두 번 적용' 되는 함정에 해당함.
        - nn.CrossEntropyLoss 는 내부적으로 log_softmax + NLLLoss 를 적용함
        - 따라서 모델 출력에 softmax 를 별도로 거치면 학습이 비정상이 됨
        자세한 설명과 비교 실험은 tutorial/44_softmax_crossentropy.py 참고.
        """
        model = []

        model += [nn.Linear(512*7*7, 4096),
                  nn.ReLU(inplace=True),
                  nn.Dropout()]
        model += [nn.Linear(4096, 4096),
                  nn.ReLU(inplace=True),
                  nn.Dropout()]
        model += [nn.Linear(4096, self.num_classes),
                  nn.Softmax(dim=1)]   # ⚠️ CrossEntropyLoss 와 이중 적용 — 함정 (44 챕터 참고)

        self.classifier = nn.Sequential(*model)

    def forward(self, x):
        """
        forward propagation 함수
        Args:
            x : torch.Tensor
                입력 데이터
        Returns:
            x : torch.Tensor
                출력 데이터
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)   
        x = self.classifier(x)
        return x


def define_network(opt, state_dict=None, device=None):
    """
    네트워크를 정의하는 함수
    Args:
        opt : argparse.Namespace
            옵션 객체
        state_dict : dict, default=None
            네트워크의 가중치 값
        device : torch.device, default=None
            device 정보
    Returns:
        network : nn.Module
            네트워크
    """
    network = CustomNetwork(opt.in_channels, opt.num_classes)

    if state_dict is not None:
        network.load_state_dict(state_dict)
    else :
        network = init_network(network, init_type='normal', init_gain=0.02)

    if device is not None:
        network = network.to(device)

    return network


def init_network(network, init_type='normal', init_gain=0.02):
    """
    네트워크의 가중치를 초기화하는 함수
    Args:
        net : nn.Module
            네트워크
        init_type : str, default='normal'
            초기화 방법
        init_gain : float, default=0.02
            초기화 gain
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    network.apply(init_func)
    return network


def define_criterion(opt):
    """
    loss 함수를 정의하는 함수
    Args:
        opt : argparse.Namespace
            옵션 객체
    Returns:
        criterion : nn.Module
            loss 함수
    """
    criterion = nn.CrossEntropyLoss()
    return criterion


def define_optimizer(network, opt):
    """
    optimizer를 정의하는 함수
    Args:
        network : nn.Module
            네트워크
        opt : argparse.Namespace
            옵션 객체
    Returns:
        optimizer : torch.optim
            optimizer
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr)
    return optimizer
