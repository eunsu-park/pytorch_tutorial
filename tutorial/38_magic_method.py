# 38_magic_method.py
# Magic Method — PyTorch 의 nn.Module 도 이 매직 메서드들을 활용한다
#
# 다음 챕터들과의 연결
#   - __init__ : 모델 정의 (37_all_layer_2.py 의 MyModel.__init__)
#   - __call__ : model(x) 호출 시 forward 가 자동 실행되는 이유
#   - __len__, __getitem__ : Dataset 클래스가 구현해야 하는 메서드 (41 챕터에서 본격적으로)
#   - __repr__ : print(model) 결과를 보기 좋게 만드는 메서드

class CustomL2UString:

    def __init__(self, string):
        self.original = string

    def __len__(self):
        return len(self.original)
    
    def __getitem__(self, idx):
        if idx >= len(self) or idx < -len(self):
            raise IndexError("CustomL2UString index out of range")
        return self.original[idx]
    
    def __repr__(self):
        return f"CustomL2UString, original: {self.original}, length: {len(self)}"
    
    def __call__(self):
        return self.original.upper()

string = "Hello World"
custom_string = CustomL2UString(string) ## __init__ 호출
print(custom_string) ## __repr__ 호출
print(len(custom_string)) ## __len__ 호출
print(custom_string[0]) ## __getitem__ 호출
print(custom_string()) ## __call__ 호출