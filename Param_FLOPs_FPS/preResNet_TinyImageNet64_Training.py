import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

"""
https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
"""
################################

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,  kernel_size=1,  stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, initial_channels, num_classes, stride=1):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,initial_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels * 8, num_blocks[3], stride=2)
        self.fc = nn.Linear(initial_channels * 8 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        fea = self.layer4(out)
        # print(fea.shape)
        out = self.avgpool(fea)
        out = out.reshape(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return fea, out

def preactresnet18(num_classes=10, dropout=False, stride=1):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], 64, num_classes, stride=stride)


def preactresnet34(num_classes=10, dropout=False, stride=1):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], 64, num_classes, stride=stride)


def preactresnet50(num_classes=10, dropout=False, stride=1):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], 64, num_classes, stride=stride)


def preactresnet101(num_classes=10, dropout=False, stride=1):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], 64, num_classes, stride=stride)


def preactresnet152(num_classes=10, dropout=False, stride=1):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], 64, num_classes, stride=stride)


def test():
    net = preactresnet34()
    _,y = net(Variable(torch.randn(1, 3, 64, 64)))
    print(y.size())


if __name__ == "__main__":
    test()
# test()

    from fvcore.nn import FlopCountAnalysis
    net = preactresnet50(200)
    net.fc = nn.Linear(2048, 200)
    # net.fc_local = nn.Linear(64, 200)
    # grid_mask = torch.randint(10, size=(64,64))
    tensor = (torch.randn(1, 3, 64, 64),) # 如果有多个参数就写成tuple形式


    flops = FlopCountAnalysis(net, tensor)
    print("FLOPs:", flops.total()/1e9)

    papa_total = sum([param.nelement() for param in net.parameters()])
    print("Param", papa_total/1e6)
