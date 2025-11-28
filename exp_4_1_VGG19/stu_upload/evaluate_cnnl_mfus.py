import os
import torch
import torch_mlu

# import torch_mlu.core.mlu_model as ct
import torch.nn as nn
import time
from load import vgg19, load_image

torch.set_grad_enabled(False)
# ct.set_device(0)

IMAGE_PATH = "data/strawberries.jpg"
VGG_PATH = "models/vgg19.pth"

if __name__ == "__main__":
    # 1. read image
    # TODO: 从指定路径加载图像
    input_image = load_image(IMAGE_PATH)
    # 2. load model
    # TODO: 生成VGG19网络模型
    net = vgg19()
    # TODO: 加载网络参数到net中
    net.load_state_dict(torch.load(VGG_PATH))
    # 3. Put our model in eval mode
    # TODO: 模型进入推理模式
    net.eval()
    # 4. jit.trace
    example_forward_input = torch.rand((1, 3, 224, 224), dtype=torch.float)
    # 5. image and model to mlu
    # TODO: 使把动态图转化为静态图
    net_traced = torch.jit.trace(net, example_forward_input)
    # TODO: 将输入图像拷贝到MLU设备
    input_image = input_image.to("mlu")
    # TODO: 将net_trace拷贝到MLU设备
    net_traced = net_traced.to("mlu")
    # 6. inference
    st = time.time()
    # TODO: 进行推理
    prob = net_traced(input_image)
    print("mlu370<cnnl backend> infer time:{:.3f} s".format(time.time() - st))
    # TODO: 将prob从MLU设备拷贝到CPU设备
    prob = prob.to("cpu")
    with open("./labels/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
        _, indices = torch.sort(prob, descending=True)
    print(
        "Classification result: id = %s, prob = %f "
        % (classes[indices[0][0]], prob[0][indices[0][0]].item())
    )
    if classes[indices[0][0]] == "strawberry":
        print("TEST RESULT PASS.")
    else:
        print("TEST RESULT FAILED.")
