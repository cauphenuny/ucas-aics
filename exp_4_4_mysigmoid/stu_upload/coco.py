from zipfile import ZipFile
from torch.utils.data import Dataset
import torch
import cv2
import numpy
import einops
import os

def find(base: str) -> str:
    for i in range(5):
        candidate = os.path.join("../" * i, base)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"cannot find {base} in parent directories, cwd={os.getcwd()}")

class COCODataSet(Dataset):

    def __init__(self):
        super(COCODataSet, self).__init__()
        self.zip_files = ZipFile(find('data/train2014_small.zip'))
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        # TODO: 使用cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成彩色图像格式。
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # TODO: 使用cv2.resize()将图像缩放为512*512大小，其中所采用的插值方式为：区域插值
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        # TODO: 使用cv2.cvtColor将图片从BGR格式转换成RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # TODO: 将image从numpy形式转换为torch.float32,并将其归一化为[0,1]
        image = torch.from_numpy(image).float() / 255.0
        # TODO: 用permute函数将tensor从HxWxC转换为CxHxW
        image = einops.rearrange(image, "h w c -> c h w")
        return image