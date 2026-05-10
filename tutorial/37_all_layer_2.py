# 37_all_layer_2.py
# Linear, Conv, Act, Dropout, BatchNorm, MaxPool 레이어를 이용한 모델 예시 - nn.Module을 상속받아 모델 구현

import torch
import torch.nn as nn

def get_num_params(model):
    """
    모델(레이어)의 파라미터 수를 계산하는 함수
    
    Args:
        model : torch.nn.Module
    Returns:
        num_params : int
    """
    return sum([p.numel() for p in model.parameters()])

class MyModel(nn.Module): # nn.Module을 상속받아 모델을 정의
    def __init__(self, in_channels, num_classes):
        super(MyModel, self).__init__() # nn.Module의 __init__() 메소드 호출
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.define_feature_extractor()
        self.define_classifier()

    def define_feature_extractor(self):
        model = []
        model += [nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)]
        model += [nn.BatchNorm2d(32)]
        model += [nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]
        model += [nn.Conv2d(32, 64, kernel_size=3, padding=1)]
        model += [nn.BatchNorm2d(64)]
        model += [nn.ReLU()]
        model += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.feature_extractor = nn.Sequential(*model)

    def define_classifier(self):
        model = []
        model += [nn.Linear(4096, 4096)]
        model += [nn.ReLU()]
        model += [nn.Dropout(p=0.5)]
        model += [nn.Linear(4096, self.num_classes)]
        model += [nn.Softmax(dim=1)]   # ⚠️ CrossEntropyLoss 와 이중 적용 함정 — tutorial/44 참고
        self.classifier = nn.Sequential(*model)

    def forward(self, x): # nn.Module을 상속받았기 때문에 forward 함수를 구현해야 모델로 작동
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model1 = MyModel(in_channels=3, num_classes=10)
print(model1)
inp1 = torch.randn(128, 3, 32, 32)
out1 = model1(inp1)
print(out1.size())
print("")

model2 = MyModel(in_channels=4, num_classes=100)
print(model2)
inp2 = torch.randn(128, 4, 32, 32)
out2 = model2(inp2)
print(out2.size())
print("")
