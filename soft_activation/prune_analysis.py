from __future__ import print_function


import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.resnet import *

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.misc import get_conv_zero_param, get_conv_zero_kernel
import matplotlib.pyplot as plt




def main():
    activation = "mish"
    model_name = "resnet18"
    num_classes = 100
    prune_ratio = [20, 36, 49, 59, 67, 74, 79, 83, 87, 89, 91, 93, 94, 95, 96, 97]
    
    for i in prune_ratio:
        model = resnet18(activation = activation, num_class=num_classes)
#         basepath = "../baseline_soft_activation/sparse_IMP/cifar100/"+ model_name +"/" +str(i)
        basepath = "sparse_IMP/cifar100/"+ model_name +"/"+ activation +"/" + str(i)
        path = basepath +"/pruned.pth.tar"
        
        
#         print("Model Path : " + path)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])

#         print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        num_parameters = sum([param.nelement() for param in model.parameters()])
        zero_parameters = get_conv_zero_param(model)
#         print('Zero parameters: {} \t Total parameters : {}'.format(zero_parameters, num_parameters))

        total= 0
        total_zero = 0
        eps = 1e-5
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                weight = m.weight.data.abs().clone()
                x, y, w, h = weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]
                zero_weight = weight.lt(eps)
                zero_weight_count_per_kernel = zero_weight.view(x, -1).sum(1)
#                 print(zero_weight.view(x, -1).shape, zero_weight_count_per_kernel.shape)
                zero_rate = zero_weight_count_per_kernel / (y * w * h)
                
                relaxed_zero_kernel_count = torch.sum(zero_rate.gt(0.95))
                total += x
                total_zero += relaxed_zero_kernel_count
#                 print("Layer : {} \t Shape : {} \t Total Kernel: {} \t Zero Kernel: {}".format(k, weight.shape, x, relaxed_zero_kernel_count))
#                 break
#         print("Total [Relaxed Zero Kernel] : %.2f" % (total_zero) )
        print("Percentage [Relaxed Zero Kernel] : %.2f" % (total_zero/total * 100) )
        
        print("------------------------------------------------------------------------------")
    
if __name__ == '__main__':
    main()
