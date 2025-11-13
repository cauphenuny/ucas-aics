# -*- coding: UTF-8 -*- 
import pycnnl
import time
import numpy as np
import os
import scipy.io
from PIL import Image
import imageio.v2 as imageio
import einops

class VGG19(object):
    def __init__(self):
        # set up net
        
        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []
        self.filter_quant_params = []

   
    def build_model(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path

        # TODO: 使用net的createXXXLayer接口搭建VGG19网络
        # creating layers
        self.net.setInputShape(1, 3, 224, 224)
        
        input_height = input_width = 224
        in_channels = 3

        def vector(*args):
            vec = pycnnl.IntVector(len(args))
            for i, val in enumerate(args):
                vec[i] = val
            data = ", ".join(str(vec[i]) for i in range(len(args)))
            print(f'{self.__class__.__name__}: Create vector with values: {data}')
            return vec

        def add_conv(layer_name, out_channels, kernel_size=3, stride=1, padding=1, relu=True):
            nonlocal in_channels, input_height, input_width
            print(f'{self.__class__.__name__}: Add conv layer {layer_name}, i={in_channels}, o={out_channels}, h={input_height}, w={input_width}')
            input_shape=vector(1, in_channels, input_height, input_width)
            # validate shapes before creating layer to catch negative dims early
            if in_channels <= 0 or input_height <= 0 or input_width <= 0:
                raise ValueError(f'Invalid shape for conv {layer_name}: in_ch={in_channels}, h={input_height}, w={input_width}')
            # createConvLayer expects (name, input_shape, in_channels, out_channels, ...)
            self.net.createConvLayer(layer_name, input_shape, out_channels, kernel_size, stride, padding, 1)
            in_channels = out_channels
            input_height = (input_height - kernel_size + 2 * padding) // stride + 1
            input_width = (input_width - kernel_size + 2 * padding) // stride + 1
            if relu:
                self.net.createReLuLayer(layer_name.replace('conv', 'relu'))
        
        def add_pool(layer_name, pool_size=2, stride=2):
            nonlocal input_height, input_width
            pool_input_shape=vector(1, in_channels, input_height, input_width)
            self.net.createPoolingLayer(layer_name, pool_input_shape, pool_size, stride)
            input_height = input_height // stride
            input_width = input_width // stride
        
        nchannels = [3, 64, 128, 256, 512, 512]
        nblocks = [0, 2, 2, 4, 4, 4]
        for layer in [1, 2, 3, 4, 5]:
            for block in range(1, nblocks[layer]+1):
                add_conv(f'conv{layer}_{block}', nchannels[layer])
            add_pool(f'pool{layer}')
        
        def add_fc(name, input_size, output_size, relu=True):
            input_shapem3=vector(1, 1, 1, input_size)
            weight_shapem3=vector(1, 1, input_size, output_size)
            output_shapem3=vector(1, 1, 1, output_size)
            self.net.createMlpLayer(name, input_shapem3,weight_shapem3,output_shapem3)
            if relu:
                self.net.createReLuLayer(name.replace('fc', 'relu'))
        
        add_fc('fc6', in_channels * input_height * input_width, 4096)
        add_fc('fc7', 4096, 4096)
        add_fc('fc8', 4096, 1000, relu=False)

        # softmax
        input_shapes=vector(1, 1, 1000)
        self.net.createSoftmaxLayer('softmax', input_shapes, 1)
    
    def load_model(self):
        # loading params ... 
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        
        count = 0
        for idx in range(self.net.size()):
            if 'conv' in self.net.getLayerName(idx):
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # TODO：调整权重形状
                # matconvnet: weights dim [height, width, in_channel, out_channel]
                # ours: weights dim [out_channel, height, width,in_channel]
                weight = einops.rearrange(weight, 'h w i o -> i h w o')
                bias = bias.reshape(-1).astype(np.float64)
                self.net.loadParams(idx, weight, bias)
                count += 1
            if 'fc' in self.net.getLayerName(idx):
                # Loading params may take quite a while. Please be patient.
                weight, bias = params['layers'][0][idx][0][0][0][0]
                weight = weight.reshape([weight.shape[0]*weight.shape[1]*weight.shape[2], weight.shape[3]])
                weight = weight.flatten().astype(np.float64)
                bias = bias.reshape(-1).astype(np.float64)
                self.net.loadParams(idx, weight, bias)
                count += 1

    def load_image(self, image_dir):
        # loading image
        self.image = image_dir
        image_mean = np.array([123.68, 116.779, 103.939])
        print('Loading and preprocessing image from ' + image_dir)
        input_image = imageio.imread(image_dir)
        pil_img = Image.fromarray(input_image)
        pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        input_image = np.array(pil_img, dtype=np.float32)
        input_image -= image_mean
        input_image = np.reshape(input_image, [1]+list(input_image.shape))
        # TODO：调整输入数据
        input_data = input_image.flatten().astype(np.float64)
        
        self.net.setInputData(input_data)

    def forward(self):
        return self.net.forward()
    
    def get_top5(self, label):
        start = time.time()
        self.forward()
        end = time.time()

        result = self.net.getOutputData()

        # loading labels
        labels = []
        with open('../synset_words.txt', 'r') as f:
            labels = f.readlines()

        # print results
        top1 = False
        top5 = False
        print('------ Top 5 of ' + self.image + ' ------')
        prob = sorted(list(result), reverse=True)[:6]
        if result.index(prob[0]) == label:
            top1 = True
        for i in range(5):
            top = prob[i]
            idx = result.index(top)
            if idx == label:
                top5 = True
            print('%f - '%top + labels[idx].strip())

        print('inference time: %f'%(end - start))
        return top1,top5
    
    def evaluate(self, file_list):
        top1_num = 0
        top5_num = 0
        total_num = 0

        start = time.time()
        with open(file_list, 'r') as f:
            file_list = f.readlines()
            total_num = len(file_list)
            for line in file_list:
                image = line.split()[0].strip()
                label = int(line.split()[1].strip())
                vgg.load_image(image)
                top1,top5 = vgg.get_top5(label)
                if top1 :
                    top1_num += 1
                if top5 :
                    top5_num += 1
        end = time.time()

        print('Global accuracy : ')
        print('accuracy1: %f (%d/%d) '%(float(top1_num)/float(total_num), top1_num, total_num))
        print('accuracy5: %f (%d/%d) '%(float(top5_num)/float(total_num), top5_num, total_num))
        print('Total execution time: %f'%(end - start))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.load_model()
    vgg.evaluate('../file_list')
