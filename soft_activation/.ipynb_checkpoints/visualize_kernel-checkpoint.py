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
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
sns.set_style("whitegrid")

parser = argparse.ArgumentParser(description='Soft Activation Visualization')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the initialization checkpoint (default: none)')
parser.add_argument('--activation', type=str, default="relu", help='Activation Function to use')
args = parser.parse_args()

def main():
    activation = args.activation
    model_name = args.model
    num_classes = 100
    prune_ratio = [20, 36, 49, 59, 67, 74, 79, 83, 87, 89, 91, 93, 94, 95, 96, 97]
    fig = plt.figure(figsize=(18,30))
    fig.subplots_adjust(hspace=0.1)
    
    
    
    path_dict = dict()
    path_dict["relu"] = "relu_mish"
    path_dict["swish"] = "swish"
    path_dict["mish"] = "mish"
    path_dict["pswish"] = "pswish_beta3"
    path_dict["gelu"] = "gelu"
  
    for i in range(0, len(prune_ratio)):
        ax = fig.add_subplot(len(prune_ratio), 3, i + 1)
        basepath = "./sparse/cifar100/"+ model_name + "/"+ path_dict[activation] +"/" +str(prune_ratio[i])
        path = basepath +"/pruned.pth.tar"
        
        if model_name == "resnet18":
            model = resnet18(activation = activation, num_class=num_classes)
            reshape_factor = 60
        elif model_name == "resnet34":
            model = resnet34(activation = activation, num_class=num_classes)
            reshape_factor = 112
        elif model_name == "resnet50":
            model = resnet50(activation = activation, num_class=num_classes)
            reshape_factor = 160
        
            
        print("Model Path : " + path)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])


        total_kernel, zero_kernel = get_conv_zero_kernel(model)
        print("Percentage [Zero Kernel] : %.2f" % (zero_kernel/total_kernel* 100) )

        
        total_kernel, zero_kernel = get_conv_zero_kernel(model)
        print("Total Number of Kernels : {}".format(total_kernel))

        conv_weights = torch.zeros(total_kernel)
        index = 0
        total= 0
        total_zero = 0
        eps = 1e-5
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                weight = m.weight.data.abs().clone()
                x, y, w, h = weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]
                zero_weight = weight.lt(eps)
                zero_weight_count_per_kernel = zero_weight.view(x, -1).sum(1)
                zero_rate = zero_weight_count_per_kernel / (y * w * h)
                
                relaxed_zero_kernel_count = torch.sum(zero_rate.gt(0.9))
                total += x
                total_zero += relaxed_zero_kernel_count
                
                size = zero_rate.numel()
                conv_weights[index:(index + size)] = zero_rate
                index += size
       
       
        conv_weights_binary = conv_weights.lt(0.9)
        
        kernel_mask = conv_weights_binary.reshape(reshape_factor, -1).T.numpy()
        print(kernel_mask.shape)
        
        colors = ["black", "whitesmoke"] 
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
        g1 = sns.heatmap(kernel_mask, cbar= False,  cmap = cmap)
        g1.set(xticklabels=[]) 
        g1.set(yticklabels=[])
        for _, spine in g1.spines.items():
            spine.set_visible(True)
        plt.ylabel(str(prune_ratio[i]), fontsize=18)
        if i == 1:
            plt.title(activation.upper()+" Kernel")
        print("-----------------------------------------------------------------------------------")
    plt.tight_layout()
#     plt.savefig(os.path.join("./visualization", model_name + "/"+ activation +'_kernel.png'))
    
    
    
if __name__ == '__main__':
    main()
