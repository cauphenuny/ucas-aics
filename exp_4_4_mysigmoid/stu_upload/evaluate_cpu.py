from torchvision.models import vgg19
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import cv2
import numpy
import time
#TODO：导入自定义动态链接库

from coco import COCODataSet
from resblock import ResBlock
from transnet import TransNet


if __name__ == '__main__':
    # TODO: 使用cpu生成图像转换网络模型并保存在g_net中
    g_net = TransNet().cpu()
    # TODO:从/models文件夹下加载网络参数到g_net中
    g_net.load_state_dict(torch.load('./models/fst.pth', map_location='cpu'))
    print("g_net build PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    batch_size = 1
    data_group = DataLoader(data_set,batch_size,True,drop_last=True)

    for i, image in enumerate(data_group):
        image_c = image.cpu()
        start = time.time()
        # TODO: 计算 g_net,得到image_g
        image_g = g_net(image_c)
        end = time.time()
        delta_time = end - start
        print("Inference (CPU) processing time: %s" % delta_time)
        #TODO: 利用save_image函数将tensor形式的生成图像image_g以及输入图像image_c以jpg格式左右拼接的形式保存在/out/cpu/文件夹下
        save_image(torch.cat([image_c, image_g], dim=3), f'./out/cpu/{i}.jpg')
        break
    print("TEST RESULT PASS!\n")



