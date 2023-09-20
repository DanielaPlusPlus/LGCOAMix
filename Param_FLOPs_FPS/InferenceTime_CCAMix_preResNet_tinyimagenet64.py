"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/clovaai/wsolevaluation/blob/master/wsol/resnet.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
import numpy as np
import math
import torch.nn.functional as F

# model_urls = { #一个维护不同模型参数下载地址的字典
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

# Returns 2D convolutional layer with space-preserving padding
"""https://github.com/amiltonwong/pytorch_fcn/blob/master/model.py"""
def conv(in_planes, out_planes, kernel_size=3, stride=1,  padding=1, output_padding=1, dilation=1, bias=False, transposed=False):
  if transposed:
    layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, bias=bias)

    # Bilinear interpolation init
    w = torch.Tensor(kernel_size, kernel_size)
    # print(layer.weight.shape, w.div(in_planes).repeat(out_planes, in_planes, 1, 1).shape)
    # torch.Size([2048, 1024, 3, 3])
    # torch.Size([1024, 2048, 3, 3])
    centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
    for y in range(kernel_size):
      for x in range(kernel_size):
        w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
    layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
  else:
    padding = (kernel_size + 2 * (dilation - 1)) // 2
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
  # if bias:
  #   init.constant(layer.bias, 0)
  return layer


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

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

        self.relu = nn.ReLU(inplace=True)
        self.conv5 = conv(512 * block.expansion, 256 * block.expansion, stride=2, transposed=True)
        self.bn5 = nn.BatchNorm2d(256 * block.expansion)
        self.conv6 = conv(256 * block.expansion, 128 * block.expansion, stride=2, transposed=True)
        self.bn6 = nn.BatchNorm2d(128 * block.expansion)
        self.conv7 = conv(128 * block.expansion, 64 * block.expansion, stride=2, transposed=True)
        self.bn7 = nn.BatchNorm2d(64 * block.expansion)
        self.conv8 = conv(64 * block.expansion, 64, kernel_size=1, stride=1, padding=0, output_padding=0,
                          transposed=True)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = conv(64, 64, kernel_size=1, stride=1, padding=0, output_padding=0, transposed=True)
        self.bn9 = nn.BatchNorm2d(64)

        self.SA = SelfAttention(input_size=64)
        self.SAP = self.SuperpixelAttentionPooling
        self.fc_local = nn.Linear(64, num_classes)

        initialize_weights(self.modules(), init_mode='xavier')

    def SuperpixelAttentionPooling(self, x, SuperP_mask, atten_top_ratio):
        # print(x.shape)#torch.Size([32, 256, 32, 32])
        topN_SurPFeat_batch = []
        topN_SurPFeat_idx_batch = []
        SPAttention_batch = []
        for sp in range(x.shape[0]):
            mask_value = np.unique(SuperP_mask[sp])
            x_sp = x[sp].reshape(x.shape[2], x.shape[3], x.shape[1])

            avgpool = []
            for v in mask_value:
                avgpool.append(x_sp[SuperP_mask[sp] == v].mean(0))
            avgpool = torch.stack(avgpool)

            avgpool_sa = self.SA(avgpool)  # get vectors

            avgpool_sa_sum = avgpool_sa.sum(1)  # vectors->num
            num = int(atten_top_ratio * (avgpool_sa_sum.shape[0]))
            if num == 0:
                num = 1
            _, map_topN_idx = torch.topk(avgpool_sa_sum, num, dim=0, largest=True)

            topN_SurPFeat_idx_batch.append(map_topN_idx)
            SPAttention_batch.append(F.sigmoid(avgpool_sa_sum))

            avgpool_sa_sp = [avgpool_sa[idx] for idx in map_topN_idx]
            avgpool_sa_sp = torch.stack(avgpool_sa_sp)
            topN_SurPFeat_batch.append(avgpool_sa_sp)

        return SPAttention_batch, topN_SurPFeat_batch, topN_SurPFeat_idx_batch

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, local=False, superpixel_map=None, topN_local_ratio=None):
        x0 = self.conv1(x)
        # x0 = self.maxpool(x)
        # print(x0.shape) #torch.Size([100, 64, 64, 64])
        # x0 = self.maxpool(x0)
        x1 = self.layer1(x0)
        # print(x1.shape)  #torch.Size([100, 64, 64, 64])
        x2 = self.layer2(x1)
        # print(x2.shape) #torch.Size([100, 128, 32, 32])
        x3 = self.layer3(x2)
        # print(x3.shape)  #torch.Size([100, 256, 16, 16])
        x4 = self.layer4(x3)
        # print(x4.shape)   #torch.Size([100, 512, 8, 8])

        """total"""
        pre_logit = self.avgpool(x4)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        # print(pre_logit.shape) #torch.Size([32, 2048])
        logits = self.fc(pre_logit)

        """local"""
        if local:
            x_up1 = self.relu(self.bn5(self.conv5(x4)))
            # print(x_up1.shape, x3.shape)  #torch.Size([32, 1024, 8, 8]) torch.Size([32, 1024, 8, 8])
            x_up2 = self.relu(self.bn6(self.conv6(x_up1 + x3)))
            # print(x_up2.shape, x2.shape)  #torch.Size([32, 512, 16, 16]) torch.Size([32, 512, 16, 16])
            x_up3 = self.relu(self.bn7(self.conv7(x_up2 + x2)))
            # print(x_up3.shape, x1.shape)  #torch.Size([32, 256, 32, 32]) torch.Size([32, 256, 32, 32])
            x_up4 = self.relu(self.bn8(self.conv8(x_up3 + x1)))
            # print(x_up4.shape, x0.shape)  #torch.Size([32, 64, 32, 32]) torch.Size([32, 64, 32, 32])
            x_up5 = self.bn9(self.conv9(x_up4 + x0))

            SPAttention_batch, topN_SurPFeat_batch, topN_SurPFeat_idx_batch = self.SAP(x_up5, superpixel_map, topN_local_ratio)  # list(32*[n, 256])

            topN_SurPLocalCls_batch = []
            for sp in range(len(topN_SurPFeat_batch)):
                x_locals_out_sp = []
                for tk in range(topN_SurPFeat_batch[sp].shape[0]):
                    x_local_out = self.fc_local(topN_SurPFeat_batch[sp][tk])
                    x_locals_out_sp.append(x_local_out)
                x_locals_out_sp = torch.stack(x_locals_out_sp)
                topN_SurPLocalCls_batch.append(x_locals_out_sp)
            """output: global_preds; selected local preds; attention for every locals; selected local index; final feature"""
            return logits, topN_SurPLocalCls_batch, topN_SurPFeat_idx_batch, SPAttention_batch,  x_up5
        else:
            return logits

    # def _make_layer(self, block, planes, blocks, stride):
    #     layers = self._layer(block, planes, blocks, stride)
    #     return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,stride)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

"""https://blog.csdn.net/beilizhang/article/details/115282604"""
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention,self).__init__()

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

        # self.out_dropout = nn.Dropout(dropout_prob)
        self.hidden_size = input_size
        self.LayerNorm = LayerNorm(input_size, eps=1e-12)

    def forward(self, input_tensor):
        """input tensor (n,d)"""
        query_layer = self.query(input_tensor)
        key_layer = self.key(input_tensor)
        value_layer = self.value(input_tensor)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.out_dropout(attention_probs)

        hidden_states = torch.matmul(attention_probs, value_layer)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

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


# from fvcore.nn import FlopCountAnalysis
# net = preactresnet50(200)
# net.fc = nn.Linear(2048, 200)
# net.fc_local = nn.Linear(64, 200)
# grid_mask = torch.randint(10, size=(64,64))
# tensor = (torch.randn(1, 3, 64, 64),True, grid_mask,0.7) # 如果有多个参数就写成tuple形式
#
#
# flops = FlopCountAnalysis(net, tensor)
# print("FLOPs:", flops.total()/1e9)
#
# papa_total = sum([param.nelement() for param in net.parameters()])
# print("Param", papa_total/1e6)

"""inference time"""
net = preactresnet50()
net.fc = nn.Linear(2048, 200)
net = net.cuda()

iterations = 300
random_input = torch.randn(1, 3, 64, 64).cuda()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = net(random_input)

# 测速
times = torch.zeros(iterations)
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = net(random_input)
        ender.record()

        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)  # 计算时间
        times[iter] = curr_time
        print(curr_time)

mean_time = times.mean().item()
print("Inference time:{:.6f}, FPS: {}".format(mean_time, 1000 / mean_time))
#inference的单位是ms(millisecond),FPS, Frame per second