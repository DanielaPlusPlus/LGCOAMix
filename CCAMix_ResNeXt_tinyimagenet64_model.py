"""
https://github.com/bearpaw/pytorch-classification/blob/master/models/imagenet/resnext.py
https://mycuhk-my.sharepoint.com/personal/1155056070_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2F1155056070%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Frelease%2Fpytorch%2Dclassification%2Fcheckpoints%2Fimagenet
"""
from __future__ import division
""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
import from https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torch
from collections import OrderedDict

__all__ = ['resnext50', 'resnext101', 'resnext152']

class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, baseWidth, cardinality, layers, num_classes):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        # 修改1
        # self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)#修改前
        self.conv1_3k3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)#修改后
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.conv5 = conv(512 * block.expansion, 256 * block.expansion, stride=2, transposed=True)
        self.bn5 = nn.BatchNorm2d(256 * block.expansion)
        self.conv6 = conv(256 * block.expansion, 128 * block.expansion, stride=2, transposed=True)
        self.bn6 = nn.BatchNorm2d(128 * block.expansion)
        self.conv7 = conv(128 * block.expansion, 64 * block.expansion, stride=2, transposed=True)
        self.bn7 = nn.BatchNorm2d(64 * block.expansion)
        self.conv8 = conv(64 * block.expansion, 64, kernel_size=1, stride=1, padding=0, output_padding=0, transposed=True)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = conv(64, 64, kernel_size=1, stride=1, padding=0, output_padding=0, transposed=True)
        self.bn9 = nn.BatchNorm2d(64)

        self.SA = SelfAttention(input_size=64)
        self.SAP = self.SuperpixelAttentionPooling
        self.fc_local = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

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
                avgpool.append(x_sp[SuperP_mask[sp]==v].mean(0))
            avgpool = torch.stack(avgpool)

            avgpool_sa = self.SA(avgpool) #get vectors

            avgpool_sa_sum = avgpool_sa.sum(1) #vectors->num
            num = int(atten_top_ratio*(avgpool_sa_sum.shape[0]))
            if num==0:
                num =1
            _, map_topN_idx = torch.topk(avgpool_sa_sum, num, dim=0, largest=True)

            topN_SurPFeat_idx_batch.append(map_topN_idx)
            SPAttention_batch.append(F.sigmoid(avgpool_sa_sum))

            avgpool_sa_sp = [avgpool_sa[idx] for idx in map_topN_idx]
            avgpool_sa_sp = torch.stack(avgpool_sa_sp)
            topN_SurPFeat_batch.append(avgpool_sa_sp)

        return SPAttention_batch, topN_SurPFeat_batch, topN_SurPFeat_idx_batch


    def forward(self, x, local=False, superpixel_map=None, topN_local_ratio=None):
        x = self.conv1_3k3(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        # x0 = self.maxpool(x)
        # print(x0.shape) #torch.Size([32, 64, 56, 56])
        # x = self.maxpool(x)
        x1 = self.layer1(x0)
        # print(x1.shape)  #torch.Size([32, 256, 56, 56])
        x2 = self.layer2(x1)
        # print(x2.shape) #torch.Size([32, 512, 28, 28])
        x3 = self.layer3(x2)
        # print(x3.shape)  #torch.Size([32, 1024, 14, 14])
        x4 = self.layer4(x3)
        # print(x4.shape)   #torch.Size([32, 2048, 14, 14])

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

            SPAttention_batch, topN_SurPFeat_batch, topN_SurPFeat_idx_batch = self.SAP(x_up5, superpixel_map,
                                                                                       topN_local_ratio)  # list(32*[n, 256])

            topN_SurPLocalCls_batch = []
            for sp in range(len(topN_SurPFeat_batch)):
                x_locals_out_sp = []
                for tk in range(topN_SurPFeat_batch[sp].shape[0]):
                    x_local_out = self.fc_local(topN_SurPFeat_batch[sp][tk])
                    x_locals_out_sp.append(x_local_out)
                x_locals_out_sp = torch.stack(x_locals_out_sp)
                topN_SurPLocalCls_batch.append(x_locals_out_sp)
            """output: global_preds; selected local preds; attention for every locals; selected local index; final feature"""
            return logits, topN_SurPLocalCls_batch, topN_SurPFeat_idx_batch, SPAttention_batch, x_up5
        else:
            return logits


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

def resnext50(pretrained=False, baseWidth=4, cardinality=32):#resnext50-32x4d
    """
    Construct ResNeXt-50.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 6, 3], 1000)
    pretrain_on_ImageNet = "./networks/resnext50_best.pth.tar"
    if pretrained:
        """remove `module.`"""
        model_dict = torch.load(pretrain_on_ImageNet)
        new_state_dict = OrderedDict()
        for k, v in model_dict['state_dict'].items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        """load only for the module names in prerained model`"""
        state = model.state_dict()
        for key in state.keys():
            if key in new_state_dict.keys():
                state[key] = new_state_dict[key]
        model.load_state_dict(state)

    return model


def resnext101(baseWidth, cardinality):
    """
    Construct ResNeXt-101.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 23, 3], 1000)
    return model


def resnext152(baseWidth, cardinality):
    """
    Construct ResNeXt-152.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 8, 36, 3], 1000)
    return model