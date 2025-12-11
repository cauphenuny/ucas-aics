from torch import nn

class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            # TODO: 进行卷积，输入通道为c，卷积核为3*3，步长为1，填充为1
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
            # TODO: 执行实例归一化
            nn.InstanceNorm2d(c),
            # TODO: 执行ReLU
            nn.ReLU(inplace=True),
            # TODO: 进行卷积，输入通道为c，卷积核为3*3，步长为1，填充为1
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
            # TODO: 执行实例归一化
            nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        # TODO: 返回残差运算的结果
        return self.layer(x) + x