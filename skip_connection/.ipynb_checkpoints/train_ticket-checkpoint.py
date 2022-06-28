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
from utils.misc import get_conv_zero_param
import matplotlib.pyplot as plt


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

# path to the Lottery Ticket initialization !!!
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the initialization checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    help='model architecture: '  +
                         ' (default: resnet18)')
parser.add_argument('--activation', type=str, default="relu", help='Activation Function to use')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 100000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    os.makedirs(args.save_dir, exist_ok=True)

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
    devset = dataloader(root='./data', train=True, download=False, transform=transform_test)  # without augmentation
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)

    num_train = len(trainset)  # should be 50000
    indices = list(range(num_train))
    split = int(0.1 * num_train)  #

    train_idx, dev_idx = indices[split:], indices[:split]  # 45000, 5000

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
            model_ref = resnet18(activation = args.activation, num_class=num_classes)
        elif args.depth == 34:
            model = resnet34(activation = args.activation, num_class=num_classes)
            model_ref = resnet34(activation = args.activation, num_class=num_classes)
        elif args.depth == 50:
            model = resnet50(activation = args.activation, num_class=num_classes)
            model_ref = resnet50(activation = args.activation, num_class=num_classes)
        elif args.depth == 101:
            model = resnet101(activation = args.activation, num_class=num_classes)
            model_ref = resnet101(activation = args.activation, num_class=num_classes)
        elif args.depth == 152:
            model = resnet152(activation = args.activation, num_class=num_classes)
            model_ref = resnet152(activation = args.activation, num_class=num_classes)
        # if not specify, the default is ResNet 18
    else:
        model = resnet18(activation = args.activation, num_class=num_classes)
        model_ref = resnet18(activation = args.activation, num_class=num_classes)
        
    model.cuda()  # model to train
    model_ref.cuda()  # pruned model
    print(model, model_ref)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # ############################### Optimizer and Loss ###############################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # ############################### Resume ###############################
    # load pruned model (model_ref), use it to mute some weights of model
    title = args.dataset + " - " + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Getting reference model from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = args.start_epoch
        model_ref.load_state_dict(checkpoint['state_dict'])

    logger = Logger(os.path.join(args.save_dir, 'log_scratch.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # set some weights to zero, according to model_ref ---------------------------------
    # ############## load Lottery Ticket (initialization parameters of un pruned model) ##############
    if args.model:
        print('==> Loading init model from %s' % args.model)
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])

    for m, m_ref in zip(model.modules(), model_ref.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m_ref.weight.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            m.weight.data.mul_(mask)

    # ############################### Train and val ###############################
    all_result = {}
    all_result['train_acc'] = []
    all_result['val_acc'] = []
    all_result['test_acc'] = []
    best_model_epoch = -1
    
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        num_parameters = get_conv_zero_param(model)
        print('Zero parameters: {}'.format(num_parameters))
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print('Parameters: {}'.format(num_parameters))

        # train model
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)

        # ######## acc on validation data each epoch ########
        dev_loss, dev_acc = test(devloader, model, criterion, epoch, use_cuda)
        
        # ######## acc on test data each epoch ########
        test_loss, test_acc = test(testloader, model, criterion, 0, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, dev_loss, train_acc, dev_acc])

        # save model after one epoch
        # Note: save all models after one epoch, to help find the best rewind
        is_best = dev_acc > best_acc
        if is_best:
            best_model_epoch = epoch
        best_acc = max(dev_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': dev_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.save_dir)
        
        # ############################### Plotting code ###############################
        
        all_result['train_acc'].append(train_acc.item())
        all_result['val_acc'].append(dev_acc.item())
        all_result['test_acc'].append(test_acc.item())
        
        fig, ax = plt.subplots(figsize=(10,6))
        plt.style.use('default')
        plt.plot(all_result['train_acc'], label='Train Acc.', color = "crimson")
        plt.plot(all_result['val_acc'], label='Val Acc.', color = "lightcoral")
        plt.plot(all_result['test_acc'], label='Test Acc.', color = "mediumseagreen")
        plt.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        plt.grid(axis = 'y')
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    print('Best Validation Accuracy: {} \t||\t Epoch: {}'.format(best_acc, best_model_epoch))
    print(best_acc)

    # ################################### test ###################################
    print('Load best model ...')
    checkpoint = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, test_acc = test(testloader, model, criterion, 0, use_cuda)
    logger.append([state['lr'], -1, test_loss, -1, test_acc])
    print('test acc (best val acc)')
    print(test_acc)

    print('Load last model ...')
    checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, test_acc = test(testloader, model, criterion, 0, use_cuda)
    logger.append([state['lr'], -1, test_loss, -1, test_acc])
    print('test acc (last epoch)')
    print(test_acc)

    logger.close()


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
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
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda):
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

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
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

    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
