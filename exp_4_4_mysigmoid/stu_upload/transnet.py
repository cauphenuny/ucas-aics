from torch import nn

import os, torch, ctypes
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
ctypes.CDLL(os.path.join(torch_lib, "libc10.so"))
import mysigmoid
from resblock import ResBlock

class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            ###################下采样层################
            # TODO：构建图像转换网络，第一层卷积
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4, bias=False),
            # TODO：实例归一化
            nn.InstanceNorm2d(32),
            # TODO：创建激活函数ReLU
            nn.ReLU(inplace=True),
            # TODO：第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            # TODO：实例归一化
            nn.InstanceNorm2d(64),
            # TODO：创建激活函数ReLU
            nn.ReLU(inplace=True),
            # TODO：第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            # TODO：实例归一化
            nn.InstanceNorm2d(128),
            # TODO：创建激活函数ReLU
            nn.ReLU(inplace=True),
            ##################残差层##################
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ################上采样层##################
            # TODO: 使用torch.nn.Upsample对特征图进行上采样
            nn.Upsample(scale_factor=2, mode="nearest"),
            # TODO: 执行卷积操作
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # TODO: 实例归一化
            nn.InstanceNorm2d(64),
            # TODO: 执行ReLU操作
            nn.ReLU(inplace=True),
            # TODO: 使用torch.nn.Upsample对特征图进行上采样
            nn.Upsample(scale_factor=2, mode="nearest"),
            # TODO: 执行卷积操作
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # TODO: 实例归一化
            nn.InstanceNorm2d(32),
            # TODO: 执行ReLU操作
            nn.ReLU(inplace=True),
            ###############输出层#####################
            # TODO: 执行卷积操作
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4, bias=True),
            # TODO: sigmoid激活函数
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # x = nn.functional.pad(x, [10, 10, 10, 10])
        # return self.layer(x)[:,:,10:-10,10:-10]
        logits = self.layer(x)
        return mysigmoid.sigmoid(logits)