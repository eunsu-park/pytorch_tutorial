import os, glob
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
from skimage.transform import resize
import numpy as np


class LoadData:
    """
    이미지 파일을 불러오고, flare class를 확인하는 클래스
    """
    def __call__(self, filepath):
        """
        이미지 파일을 불러오고, flare class를 확인하는 함수
        Args:
            filepath : str
                이미지 파일 경로
        Returns:
            image : numpy.ndarray
                불러온 이미지 데이터
            flare_class : str
                flare class
        """
        image = imread(filepath)
        flare_class = filepath.split(os.sep)[-2]
        return image, flare_class


class ResizeData:
    """
    이미지 크기를 조절하는 클래스
    """
    def __init__(self, image_size=224):
        """
        ResizeData 클래스의 생성자
        Args:
            image_size : int or tuple
                이미지 크기
        """
        if isinstance(image_size, int) :
            image_size = (image_size, image_size)
        self.image_size = image_size

    def __call__(self, image):
        """
        이미지 크기를 조절하는 함수
        Args:
            image : numpy.ndarray
                이미지 데이터
        Returns:
            image : numpy.ndarray
                크기가 조절된 이미지 데이터
        """
        image = resize(image, self.image_size, order=1, mode="constant", preserve_range=True)
        if len(image.shape) == 2 :
            image = np.expand_dims(image, axis=0)
        return image


class NormalizeData:
    """
    이미지 데이터를 0~1 사이로 정규화하는 클래스
    """
    def __call__(self, image) :
        """
        이미지 데이터를 0~1 사이로 정규화하는 함수
        Args:
            image : numpy.ndarray
                이미지 데이터
        Returns:
            image : numpy.ndarray
                정규화된 이미지 데이터
        """
        return image/255.


class MakeLabel:
    """
    flare class를 one-hot encoding 형태의 label로 변환하는 클래스
    """
    def __init__(self, flare_classes=['C', 'M', 'X'], non_flare_classes=['B', 'A', 'Non']):
        """
        MakeLabel 클래스의 생성자
        Args:
            flare_classes : list
                flare class 리스트
            non_flare_classes : list
                flare class가 아닌 클래스 리스트
        """
        self.flare_classes = flare_classes
        self.non_flare_classes = non_flare_classes
            
    def __call__(self, flare_class):
        """
        flare class를 one-hot encoding 형태의 label로 변환하는 함수
        Args:
            flare_class : str
                flare class
        Returns:
            label : list
                one-hot encoding 형태의 label
        """
        if flare_class in self.flare_classes :
            label = [0., 1.]
        elif flare_class in self.non_flare_classes :
            label = [1., 0.]
        else :
            raise ValueError(f"Invalid flare class : {flare_class}")
        return label


class ToTensor:
    """
    numpy array를 torch tensor로 변환하는 클래스
    """
    def __init__(self, dtype=torch.float32):
        """
        ToTensor 클래스의 생성자
        Args:
            dtype : torch.dtype
                torch tensor의 데이터 타입
        """
        self.dtype=dtype
    def __call__(self, data):
        """
        numpy array를 torch tensor로 변환하는 함수
        Args:
            data : numpy.ndarray
                데이터
        Returns:
            data : torch.Tensor
                변환된 데이터
        """
        data = torch.tensor(data, dtype=self.dtype)
        return data


class CustomDataset(Dataset):
    """
    flare 데이터셋을 불러오는 클래스
    """
    def __init__(self, data_root, image_size, is_train=True):
        """
        CustomDataset 클래스의 생성자
        Args:
            data_root : str
                flare 데이터셋의 경로
            image_size : int or tuple
                이미지 크기
            is_train : bool
                학습 데이터셋인지 테스트 데이터셋인지 여부
        """
        if is_train is True :
            pattern = f"{data_root}/Train/*/*/*.png"
        else :
            pattern = f"{data_root}/Test/*/*/*.png"
        self.list_data = glob.glob(pattern)
        self.nb_data = len(self.list_data)
        self.normalize = NormalizeData
        self.resize = ResizeData(image_size)
        self.totensor = ToTensor()
        
    def __len__(self):
        """
        데이터셋의 크기를 반환하는 함수
        Returns:
            int: 데이터셋의 크기
        """
        return self.nb_data
    
    def __getitem__(self, idx):
        """
        데이터셋의 idx번째 데이터를 반환하는 함수
        Args:
            idx : int
                데이터 인덱스
        Returns:
            image : torch.Tensor
                이미지 데이터
            label : torch.Tensor
                레이블 데이터
        """
        image, flare_class = LoadData()(self.list_data[idx])
        image = self.normalize(image)
        image = self.resize(image)
        image = self.totensor(image)

        label = MakeLabel()(flare_class)
        label = self.totensor(label)
        return image, label


def define_dataset(opt):
    """
    dataset과 dataloader를 정의하는 함수
    Args:
        opt : argparse.ArgumentParser
            사용자 입력값
    Returns:
        dataset : CustomDataset
            데이터셋 객체
        dataloader : DataLoader
            데이터로더 객체
    """
    dataset = CustomDataset(opt.data_root, opt.image_size, opt.is_train)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=opt.is_train, num_workers=opt.num_workers)
    return dataset, dataloader
