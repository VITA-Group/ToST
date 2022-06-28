'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation = "relu"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = activation_list[activation](inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.conv_jump1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,  bias=False)
        self.conv_jump2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1,  bias=False)
        self.conv_jump1_bn = nn.BatchNorm2d(planes)
        self.conv_jump2_bn = nn.BatchNorm2d(planes)
        
        self.conv_jump1.weight.data.fill_(0.0001)
        self.conv_jump2.weight.data.fill_(0.0001)
    
    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = out1 + self.conv_jump1_bn(self.conv_jump1(x))
        out2 = self.activation(out1)
        
        
        out3 = self.bn2(self.conv2(out2))
        out4 = out3 + self.shortcut(x)
        out4 = out4 + self.conv_jump2_bn(self.conv_jump2(out2))
        out5 = self.activation(out4)
        return out5


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation = "relu"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.activation = activation_list[activation](inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, activation = "relu", num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], activation, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], activation, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], activation, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], activation, stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.activation = activation_list[activation](inplace=True)
        
    def _make_layer(self, block, planes, num_blocks, activation, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18(activation = "relu", num_class=10):
    print(activation)
    return ResNet(BasicBlock, [2,2,2,2], activation, num_classes=num_class)


def resnet34(activation = "relu", num_class=10):
    return ResNet(BasicBlock, [3,4,6,3], activation, num_classes=num_class)


def resnet50(activation = "relu", num_class=10):
    return ResNet(Bottleneck, [3,4,6,3], activation, num_classes=num_class)


def resnet101(activation = "relu", num_class=10):
    return ResNet(Bottleneck, [3,4,23,3], activation, num_classes=num_class)


def resnet152(activation = "relu", num_class=10):
    return ResNet(Bottleneck, [3,8,36,3], activation, num_classes=num_class)


if __name__ == "__main__":
    def test():
        net = resnet18()
        y = net(torch.randn(1, 3, 32, 32))
        print(y.size())


    test()

