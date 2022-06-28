from __future__ import print_function


import argparse
import os
import random
import shutil
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary

from models.resnet import *

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import matplotlib.pyplot as plt
from activations import *

activation_list = {'relu': nn.ReLU,
                   'swish': nn.SiLU,
                   'softplus': nn.Softplus,
                   'elu': nn.ELU,
                   'pswish' : SwishParameteric,
                   'mish' : Mish,
                   'gelu' : GeLU,
                   'lisht' : LiSHT,
}

# ############################### Parameters ###############################
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=182, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[90, 135],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default=None, type=str, help='pretrained model')
parser.add_argument('--eval', action="store_true", help="evaluation pretrained model")

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' (default: resnet18)')
parser.add_argument('--activation', type=str, default="relu", help='Activation Function to use')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--layer', type=int, default=16, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

# Visualize feature maps
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    
    # ############################### Dataset ###############################
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100

    # #################### train, dev, test split ####################
    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)  # with augmentation
    devset = dataloader(root='./data', train=True, download=False, transform=transform_test)    # without augmentation
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)

    num_train = len(trainset)               # should be 50000
    indices = list(range(num_train))
    split = int(0.1 * num_train)            #

    train_idx, dev_idx = indices[split:], indices[:split]     # 45000, 5000

    trainset = data.Subset(trainset, train_idx)
    devset = data.Subset(devset, dev_idx)

    print('Total image in train, ', len(trainset))
    print('Total image in valid, ', len(devset))
    print('Total image in test, ', len(testset))

    trainloader = data.DataLoader(trainset, batch_size=args.train_batch,
                                  shuffle=True, num_workers=args.workers)

    devloader = data.DataLoader(devset, batch_size=args.test_batch,
                                shuffle=False, num_workers=args.workers)

    testloader = data.DataLoader(testset, batch_size=args.test_batch,
                                 shuffle=False, num_workers=args.workers)

    # ############################### Model ###############################
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnet'):
        # using new Res Net architecture #####################
        if args.depth == 18:
            model = resnet18(activation = args.activation, num_class=num_classes)
        elif args.depth == 34:
            model = resnet34(activation = args.activation, num_class=num_classes)
        elif args.depth == 50:
            model = resnet50(activation = args.activation, num_class=num_classes)
        elif args.depth == 101:
            model = resnet101(activation = args.activation, num_class=num_classes)
        elif args.depth == 152:
            model = resnet152(activation = args.activation, num_class=num_classes)
    else:
        model = resnet18(activation = args.activation, num_class=num_classes)
    
#     model.conv1.register_forward_hook(get_activation('layer2.conv1'))
    model.cuda()
#     print(summary(model, (3, 32, 32)))
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    
    # ############################### Optimizer and Loss ###############################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    
    ######################### only evaluation ###################################
    if args.eval:
        assert args.pretrained
    pretrained_model = torch.load(args.pretrained)
    print('loading from state_dict')
    if 'state_dict' in pretrained_model.keys():
        pretrained_model = pretrained_model['state_dict']
    model.load_state_dict(pretrained_model)
    
    print("-"*50)
    for k, m in enumerate(model.modules()):
        if isinstance(m, activation_list[args.activation]):
#             print(k, m)
            if k == args.layer:
                print("-------------------")
                print(k, m)
                print("-------------------")
                activated_features = SaveFeatures(m)
    
    test(testloader, model, criterion, 0, use_cuda, activated_features)
    

    return
    print("Only defined for evaluation!")
    
def test(testloader, model, criterion, epoch, use_cuda, activated_features):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    
    total_activation = 0
    zero_activation = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

#             print("*"*50)
            activationMap = activated_features.features
#             print("\n")
#             print(activationMap[0][0][0])
            activationMap = (np.abs(activationMap) > 1e-2)
#             print(activationMap[0][0][0])
            total_activation += activationMap.size
            zero_activation += activationMap.size - np.count_nonzero(activationMap)
           

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
    bar.finish()
    print("Activation shape: {} || Total Activation: {} || Zero Activations: {}".format(activationMap.shape, total_activation, zero_activation)) 
    print('Sparsity :  %.2f' % (zero_activation/total_activation * 100))
    return (losses.avg, top1.avg)


if __name__ == '__main__':
    main()