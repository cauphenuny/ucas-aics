from torchvision.models import vgg19
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import torch_mlu
import cv2
import numpy
import os
from train import TransNet, VGG19, load_image

os.putenv('MLU_VISIBLE_DEVICES','0')

def get_gram_matrix(f_map):
    """
    获取格拉姆矩阵
    :param f_map:特征图
    :return:格拉姆矩阵，形状（通道数,通道数）
    """
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        f_map = f_map.reshape(n, c, h * w)
        gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
        return gram_matrix


if __name__ == '__main__':
    image_style = load_image('./data/udnie.jpg').cpu()
    # TODO: 将输入的风格图像加载到mlu设备上,得到mlu_image_style
    mlu_image_style = image_style.to("mlu")
    net = VGG19().cpu()
    g_net = TransNet().cpu()
    # TODO: 将图像转换网络加载到mlu得到mlu_g_net
    mlu_g_net = g_net.to("mlu")
    # TODO: 将特征网络加载到mlu得到mlu_net
    mlu_net = net.to("mlu")
    print("mlu_net build PASS!\n")
    # TODO: 使用adam优化器对mlu_g_net的参数进行优化
    optimizer = torch.nn.optim.Adam(mlu_g_net.parameters(), lr=0.001)
    # TODO: 在cpu上计算均方误差损失函得到loss_func
    loss_func = torch.nn.MSELoss()
    # TODO: 将损失函数加载到mlu上得到mlu_loss_func
    mlu_loss_func = loss_func.to("mlu")
    print("build loss PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    batch_size = 1
    data_loader = DataLoader(data_set, batch_size, True, drop_last=True)
    # TODO：mlu_iamge_style经过特征提取网络mlu_net生成风格特征s1-s4
    s1, s2, s3, s4 = mlu_net(mlu_image_style)
    # TODO: 对风格特征s1-s4计算格拉姆矩阵并从当前计算图中分离下来，得到对应的s1-s4
    s1 = get_gram_matrix(s1).detach()
    s2 = get_gram_matrix(s2).detach()
    s3 = get_gram_matrix(s3).detach()
    s4 = get_gram_matrix(s4).detach()
    j = 0
    count = 0
    epochs = 0
    while j <= epochs:
        for i, image in enumerate(data_loader):
            """生成图片，计算损失"""
            image_c = image.cpu()
            # TODO: 将输入图像拷贝到mlu上得到mlu_image_c
            mlu_image_c = image_c.to("mlu")
            # TODO: 将mlu_image_c经过mlu_g_net输出生成图像mlu_imge_g
            mlu_image_g = mlu_g_net(mlu_image_c)
            # TODO: 利用特征提取网络mlu_net提取生成图像mlu_image_g的特征out1-out4
            out1, out2, out3, out4 = mlu_net(mlu_image_g)
            """计算风格损失"""
            # TODO: 对生成图像的特征out1-out4计算gram矩阵，并与风格图像的特征求损失，分别得到loss_s1-loss_s4
            loss_s1 = mlu_loss_func(get_gram_matrix(out1), s1)
            loss_s2 = mlu_loss_func(get_gram_matrix(out2), s2)
            loss_s3 = mlu_loss_func(get_gram_matrix(out3), s3)
            loss_s4 = mlu_loss_func(get_gram_matrix(out4), s4)
            # TODO：loss_s1-loss_s4相加得到风格损失loss_s
            loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4

            """计算内容损失"""
            # TODO: 将图片mlu_image_c经过特征提取网络mlu_net得到内容特图像的特征c1-c4
            c1, c2, c3, c4 = mlu_net(mlu_image_c)
            # TODO: 将内容图像特征c2从计算图中分离并与内容图像特征out2求内容损失loss_c2
            loss_c2 = mlu_loss_func(c2.detach(), out2)
            loss_c = loss_c2

            """总损失"""
            loss = loss_c + 0.000000005 * loss_s

            """清空梯度、计算梯度、更新参数"""
            # TODO: 梯度初始化为零
            optimizer.zero_grad()
            # TODO: 反向传播求梯度
            loss.backward()
            # TODO: 更新所有参数
            optimizer.step()
            print('j:',j, 'i:',i, 'loss:',loss.item(), 'loss_c:',loss_c.item(), 'loss_s:',loss_s.item())
            count += 1
            mlu_image_g = mlu_image_g.cpu()
            mlu_image_c = mlu_image_c.cpu()
            if i % 10 == 0:
                # TODO: 将图像转换网络fst_train_mlu.pth的参数存储在models/文件夹下
                torch.save(mlu_g_net.state_dict(), "models/fst_train_mlu.pth")
                # TODO: 利用save_image函数将tensor形式的生成图像mlu_image_g以及输入图像mlu_image_c以jpg左右拼接的形式保存在/out/train_mlu/文件夹下
                save_image(
                    torch.cat((mlu_image_g, mlu_image_c), dim=3),
                    f"/out/train_mlu/{j}_{i}.jpg",
                )
        j += 1

print("MLU TRAIN RESULT PASS!\n")
