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
        모델 출력은 logit. nn.CrossEntropyLoss 는 내부적으로 log_softmax 를 적용하므로
        모델 마지막에 nn.Softmax 를 두지 않는다.
        """
        model = []

        model += [nn.Linear(512*7*7, 4096),
                  nn.ReLU(inplace=True),
                  nn.Dropout()]
        model += [nn.Linear(4096, 4096),
                  nn.ReLU(inplace=True),
                  nn.Dropout()]
        model += [nn.Linear(4096, self.num_classes)]

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
