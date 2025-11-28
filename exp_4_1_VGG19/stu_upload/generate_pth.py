import os
import scipy.io
import torch
import torch.nn as nn
from collections import OrderedDict
from load import vgg19

os.putenv("MLU_VISIBLE_DEVICES", "")
IMAGE_PATH = "data/strawberries.jpg"
VGG_PATH = "data/imagenet-vgg-verydeep-19.mat"


if __name__ == "__main__":
    # TODO:使用scipy加载.mat格式的VGG19模型
    datas = scipy.io.loadmat(VGG_PATH)
    model = vgg19()
    new_state_dict = OrderedDict()
    for i, param_name in enumerate(model.state_dict()):
        name = param_name.split(".")
        if name[-1] == "weight":
            new_state_dict[param_name] = torch.from_numpy(datas[str(i)]).float()
        else:
            new_state_dict[param_name] = torch.from_numpy(datas[str(i)][0]).float()
    # TODO:加载网络参数到model
    model.load_state_dict(new_state_dict)
    print("*** Start Saving pth ***")
    # TODO:保存模型的参数到models/vgg19.pth
    torch.save(model.state_dict(), "models/vgg19.pth")
    print("Saving pth  PASS.")
