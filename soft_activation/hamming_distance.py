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
import pandas as pd

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.misc import get_conv_zero_param, get_conv_zero_kernel
import matplotlib.pyplot as plt

import seaborn as sns

parser = argparse.ArgumentParser(description='Soft Activation Visualization')
parser.add_argument('--act1', type=str, default="relu", help='Activation Function to use')
parser.add_argument('--act2', type=str, default="relu", help='Activation Function to use')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the initialization checkpoint (default: none)')
args = parser.parse_args()

def parameter_count(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()
    return total

def get_mask1D(model):
    total = parameter_count(model)
    mask = torch.zeros(total)
    index = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weight = m.weight.data.abs().clone()
            weight_mask = weight.eq(0)
            size = weight.numel()
            mask[index:(index + size)] = weight_mask.view(-1)
            index += size
    return mask

def main():
    activation1 = args.act1
    activation2 = args.act2
    model_name = args.model
    num_classes = 100
    fig = plt.figure(figsize=(8,16))
    fig.subplots_adjust(hspace=0.12)
    
    prune_ratio = [20, 36, 49, 59, 67, 74, 79, 83, 87, 89, 91, 93, 94, 95, 96, 97]
#     prune_ratio = [20, 36, 49, 59]
    distance = torch.zeros(len(prune_ratio), len(prune_ratio))
    
    
    path_dict = dict()
    path_dict["relu"] = "relu_mish"
    path_dict["swish"] = "swish"
    path_dict["mish"] = "mish"
    path_dict["pswish"] = "pswish_beta3"
    
    basepath = "./sparse/cifar100/"+ model_name + "/{}/{}" 
    path = basepath +"/pruned.pth.tar"
        
    for i in range(0, len(prune_ratio)):
        for j in range(0, len(prune_ratio)):
            if model_name == "resnet18":
                model1 = resnet18(activation = activation1, num_class=num_classes)
                model2 = resnet18(activation = activation2, num_class=num_classes)
            elif model_name == "resnet34":
                model1 = resnet34(activation = activation1, num_class=num_classes)
                model2 = resnet34(activation = activation2, num_class=num_classes)

            print("=> Loading {}".format(path.format(path_dict[activation1], prune_ratio[i])))
            checkpoint = torch.load(path.format(path_dict[activation1], prune_ratio[i]))
            model1.load_state_dict(checkpoint['state_dict'])

            print("=> Loading {}".format(path.format(path_dict[activation2], prune_ratio[j])))
            checkpoint = torch.load(path.format(path_dict[activation2], prune_ratio[j]))
            model2.load_state_dict(checkpoint['state_dict'])

            mask_model1 = get_mask1D(model1)
            mask_model2 = get_mask1D(model2)

            dist = torch.sum(mask_model1 != mask_model2)
            print(dist)
            print("-"*40)
            distance[i , j] = dist
    
    ax = fig.add_subplot(2, 1, 1)
    g1 = sns.heatmap(distance, cbar= True,  cmap = 'gray')
    g1.set(xticklabels=prune_ratio) 
    g1.set(yticklabels=prune_ratio)
    # make frame visible
    for _, spine in g1.spines.items():
        spine.set_visible(True)
        
    plt.title("Mask Distance({}, {})".format(activation1, activation2))
    ax2 = fig.add_subplot(2, 1, 2)
    diagonal = torch.diagonal(distance, 0)
    print(diagonal)
    df = pd.DataFrame()
    df["prune_ratio"] = prune_ratio
    df["distance"] = list(diagonal.numpy())
    g2 = sns.barplot(x="prune_ratio", y="distance", data=df)
    
    
    plt.savefig(os.path.join("./visualization", model_name + "/"+ activation1 + "_" + activation2 +'_distance.png'))
    
if __name__ == '__main__':
    main()

            
    
            
    