# 14_magic_method.py
# Magic Method — PyTorch 의 nn.Module 과 Dataset 을 이해하기 위한 사전 학습
#
# 다음 챕터들과의 연결
#   - __init__              : 모델/데이터셋 초기화 (16, 25, 26 챕터)
#   - __call__              : model(x) 호출 시 forward 가 자동 실행되는 이유 (25)
#   - __len__, __getitem__  : Dataset 클래스가 구현해야 하는 메서드 (26)
#   - __repr__              : print(model) 결과를 보기 좋게 만드는 메서드


# ────────────── 사용자 정의 클래스에 매직 메서드 구현 ──────────────
class CustomString:
    """문자열을 감싸는 간단한 컨테이너 — 매직 메서드 시연용"""

    def __init__(self, string):
        # 인스턴스를 만들 때 호출됨 : CustomString("Hello")
        self.original = string

    def __len__(self):
        # len(obj) 가 호출되면 실행됨
        return len(self.original)

    def __getitem__(self, idx):
        # obj[idx] 또는 for x in obj 가 호출되면 실행됨
        if idx >= len(self) or idx < -len(self):
            raise IndexError("CustomString index out of range")
        return self.original[idx]

    def __repr__(self):
        # print(obj) 또는 repr(obj) 가 호출되면 실행됨
        return f"CustomString(original={self.original!r}, length={len(self)})"

    def __call__(self):
        # obj() 처럼 함수처럼 호출하면 실행됨
        return self.original.upper()


# ────────────── 동작 확인 ──────────────
s = CustomString("Hello World")     # __init__
print(s)                            # __repr__
print(f"len(s)   = {len(s)}")       # __len__
print(f"s[0]     = {s[0]}")         # __getitem__
print(f"s()      = {s()}")          # __call__

# 반복 — __getitem__ 이 정수 인덱스를 받기 때문에 for 문도 자동 동작
print("for 문 : ", end="")
for ch in s:
    print(ch, end="")
print("")
print("")


# ────────────── PyTorch 와의 연결 ──────────────
# 실제 PyTorch 클래스가 매직 메서드를 어떻게 활용하는지 미리보기 :
#
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, ...):  # 데이터 경로/리스트 저장
#     def __len__(self):        # 전체 샘플 수
#     def __getitem__(self, i): # i 번째 샘플 반환  ← DataLoader 가 자동 호출
#
# class MyModel(torch.nn.Module):
#     def __init__(self):  # 레이어 등록 (super().__init__() 필수)
#     def forward(self, x):# 순전파 정의
# model = MyModel()
# y = model(x)            ← __call__ 이 forward 를 자동 호출
#
# print(model)            ← __repr__ 가 레이어 구조를 예쁘게 출력

# ────────────── 비교 정리 ──────────────
# 매직 메서드               역할                              PyTorch 에서의 사용처
# ───────────────────────  ───────────────────────────────  ────────────────────────
# __init__                  객체 생성 시 초기화                Module 의 레이어 등록
# __call__                  obj() 호출 시 동작 정의            model(x) → forward(x)
# __len__                   len(obj)                          len(dataset)
# __getitem__               obj[idx]                          dataset[i]  /  DataLoader 의 인덱싱
# __repr__                  print(obj) / repr(obj)            print(model) 의 트리 출력
