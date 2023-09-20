import torchvision
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch
import math
import numpy as np
import copy
import itertools
import torch.nn.functional as F
from torch.nn import init
"""
x1和x2,x_mix一起输入,x1和x2进行local classification, loss值对应entropy,每个cell, loss越小,交叉熵越小, 能代表target的可能性越大.mix时候取到的概率更大
"""
"""
https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
"""
################################

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],**kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet18'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    ##load model trained from imagenet
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)

    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    ##load model trained from imagenet
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet101'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)

    return model

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 网络输入部分
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1_3k3 = nn.Conv2d(3, 64, kernel_size=3,padding=1, bias=False)#修改点1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 中间卷积部分
        # self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer1_3k3 = self._make_layer(block, 64, layers[0], stride=1)#修改点2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.ConvT1 = nn.ConvTranspose2d(512 * block.expansion, 256 * block.expansion, kernel_size=(2,2), stride=2, padding=0)
        # self.bnT1 = nn.BatchNorm2d(256 * block.expansion)
        # self.ConvT2 = nn.ConvTranspose2d(256 * block.expansion, 128 * block.expansion, kernel_size=(2,2), stride=2, padding=0)
        # self.bnT2 = nn.BatchNorm2d(128 * block.expansion)
        # # self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.ConvT3 = nn.ConvTranspose2d(128 * block.expansion, 64 * block.expansion, kernel_size=(2,2), stride=2, padding=0)
        # self.bnT3 = nn.BatchNorm2d(64 * block.expansion)
        # # self.ConvT4 = nn.ConvTranspose2d(64 * block.expansion, 64 * block.expansion, kernel_size=(1,1), stride=1, padding=0)
        # # self.bnT4 = nn.BatchNorm2d(64 * block.expansion)

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
        # self.conv10 = conv(32, num_classes, kernel_size=7)
        # init.constant(self.conv10.weight, 0)  # Zero init

        self.SA = SelfAttention(input_size=64)
        self.SAP = self.SuperpixelAttentionPooling

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.conv_local = nn.Conv2d(64 * block.expansion, num_classes, 3, 1, 1)
        # self.bn_local = nn.BatchNorm2d(num_classes)
        self.fc_local = nn.Linear(16 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):#是用来构建ResNet网络中的4个blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        #将每个blocks的第一个residual结构保存在layers列表中
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #将每个blocks的剩下residual 结构保存在layers列表中
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # def SuperpixelPooling(selfself,x,SuperP_mask):
    #     # print(x.shape)#torch.Size([32, 256, 32, 32])
    #     # out = torch.ones_like(x)
    #     for sp in range(x.shape[0]):
    #         mask_value = np.unique(SuperP_mask[sp])
    #         SuperP_mask_ch = SuperP_mask[sp].expand(x.shape[1],-1,-1)
    #         for v in mask_value:
    #             x[sp][SuperP_mask_ch==v] = x[sp][SuperP_mask_ch==v].mean()
    #     return x

    # def SuperpixelPooling(selfself,x,SuperP_mask):
    #     # print(x.shape)#torch.Size([32, 256, 32, 32])
    #     avg_pool_batch = []
    #     for sp in range(x.shape[0]):
    #         mask_value = np.unique(SuperP_mask[sp])
    #         x_sp = x[sp].reshape(x.shape[2], x.shape[3], x.shape[1])
    #         # print(x_sp.shape) #torch.Size([32, 32, 256])
    #         avg_pool = []
    #         for v in mask_value:
    #             # print(SuperP_mask[sp].shape) #torch.Size([32, 32])
    #             # print(x_sp[SuperP_mask[sp]==v].mean(0).shape) #torch.Size([256])
    #             avg_pool.append(x_sp[SuperP_mask[sp]==v].mean(0))
    #         avg_pool = torch.stack(avg_pool)
    #         avg_pool_batch.append(avg_pool)
    #     avg_pool_batch = torch.stack(avg_pool_batch)
    #
    #     return avg_pool_batch

    def SuperpixelAttentionPooling(self, x, SuperP_mask, atten_top_ratio):
        # print(x.shape)#torch.Size([32, 256, 32, 32])
        avgpool_sa_batch_sel = []
        avgpool_sa_batch = []
        topN_idx_batch = []
        for sp in range(x.shape[0]):
            mask_value = np.unique(SuperP_mask[sp])
            x_sp = x[sp].reshape(x.shape[2], x.shape[3], x.shape[1])

            avgpool = []
            for v in mask_value:
                avgpool.append(x_sp[SuperP_mask[sp]==v].mean(0))
            avgpool = torch.stack(avgpool)

            avgpool_sa = self.SA(avgpool)

            avgpool_sa_spacial = avgpool_sa.sum(1)
            _, map_topN_idx = torch.topk(avgpool_sa_spacial, int(atten_top_ratio*(avgpool_sa_spacial.shape[0])), dim=0, largest=True)

            topN_idx_batch.append(map_topN_idx)
            avgpool_sa_batch.append(avgpool_sa)

            if len(map_topN_idx)>0:
                avgpool_sa_sp = [avgpool_sa[idx] for idx in map_topN_idx]
                avgpool_sa_sp = torch.stack(avgpool_sa_sp)
                avgpool_sa_batch_sel.append(avgpool_sa_sp)
            else:
                avgpool_sa_batch_sel.append(avgpool_sa)

        return avgpool_sa_batch, avgpool_sa_batch_sel, topN_idx_batch

    # def SuperpixelAttentionPooling(self,x,SuperP_mask):
    #     # print(x.shape)#torch.Size([32, 256, 32, 32])
    #     avgpool_sa_batch = []
    #     topN_idx_batch = []
    #     for sp in range(x.shape[0]):
    #         mask_value = np.unique(SuperP_mask[sp])
    #         x_sp = x[sp].reshape(x.shape[2], x.shape[3], x.shape[1])
    #         # print(x_sp.shape) #torch.Size([32, 32, 256])
    #
    #         avgpool = []
    #         # print(mask_value.shape[0])
    #         for v in mask_value:
    #             # print(SuperP_mask[sp].shape) #torch.Size([32, 32])
    #             # print(x_sp.shape)
    #             # print(x_sp[SuperP_mask[sp]==v].shape) #torch.Size([x, 256])
    #
    #             #average pooling
    #             avgpool.append(x_sp[SuperP_mask[sp]==v].mean(0))
    #
    #             # # #max pooling
    #             # max_value,_ = x_sp[SuperP_mask[sp]==v].max(0)
    #             # avgpool.append(max_value)
    #
    #         avgpool = torch.stack(avgpool)
    #
    #         avgpool_sa = self.SA(avgpool)
    #         # avgpool_sa = avgpool
    #
    #         # weights_sap = F.sigmoid(avgpool_sa.sum(1))
    #         # print(F.sigmoid(avgpool_sa_spacial))
    #         # print(avgpool_sa_spacial.shape[0])
    #
    #         avgpool_sa_batch.append(avgpool_sa)
    #
    #     return avgpool_sa_batch

    def forward(self, x, local=False, superpixel_map=None, topN_local_ratio=None):
        x0 = self.relu(self.bn1(self.conv1_3k3(x)))
        # print(x0.shape) #torch.Size([32, 64, 32, 32])
        # x = self.maxpool(x)
        x1 = self.layer1_3k3(x0)
        # print(x1.shape)  #torch.Size([32, 256, 32, 32])
        x2 = self.layer2(x1)
        # print(x2.shape) #torch.Size([32, 512, 16, 16])
        x3 = self.layer3(x2)
        # print(x3.shape)  #torch.Size([32, 1024, 8, 8])
        x4 = self.layer4(x3)
        # print(x4.shape)   #torch.Size([32, 2048, 4, 4])

        """total"""
        pre_logit = self.avgpool(x4)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        # """local"""
        # if local:
        #     x_up1 = self.relu(self.bnT1(self.ConvT1(x)))      #(32,1024,8,8)
        #     x_up2 = self.relu(self.bnT2(self.ConvT2(x_up1)))  #(32,512,16,16)
        #     x_up3 = self.relu(self.bnT3(self.ConvT3(x_up2)))  #(32,256,32,32)
        #     # print(x_up1.shape)
        #     # print(x_up2.shape)
        #     # print(x_up3.shape)
        #     # print(superpixel_map[0].shape)
        #     x_locals_out_batch = []
        #     x_sap_list, topN_idx = self.SAP(x_up3, superpixel_map,atten_top_ratio)  # list(32*[n, 256])
        #
        #     for sp in range(len(x_sap_list)):
        #         # print(x_sap_list[sp].shape)
        #         x_locals_out_sp = []
        #         for tk in range(x_sap_list[sp].shape[0]):
        #             x_local_out = self.fc_local(x_sap_list[sp][tk])
        #             x_locals_out_sp.append(x_local_out)
        #         x_locals_out_sp = torch.stack(x_locals_out_sp)
        #         x_locals_out_batch.append(x_locals_out_sp)
        #     return logits, x_locals_out_batch,topN_idx
        # else:
        #     return logits

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
            # x_up6 = self.conv10(x_up5)
            # print(x_up5.shape) #torch.Size([32, 64, 32, 32])
            # print(x_up6.shape)
            # print(superpixel_map[0].shape)

            x_sap_list, x_sap_sel_list, topN_local_idx = self.SAP(x_up5, superpixel_map, topN_local_ratio)  # list(32*[n, 256])

            x_locals_out_batch = []
            weights_locals_out_batch = []
            for sp in range(len(x_sap_sel_list)):
                x_locals_out_sp = []
                for tk in range(x_sap_sel_list[sp].shape[0]):
                    x_local_out = self.fc_local(x_sap_sel_list[sp][tk])
                    x_locals_out_sp.append(x_local_out)
                x_locals_out_sp = torch.stack(x_locals_out_sp)
                x_locals_out_batch.append(x_locals_out_sp)
                weights_locals_out_batch.append(F.sigmoid(x_sap_list[sp].sum(1)))
            return logits, x_locals_out_batch, weights_locals_out_batch, topN_local_idx
        else:
            return logits

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample   #对输入特征图大小进行减半处理
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


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

    # def forward(self, input_tensor):
    #     """input tensor (bsz,c,w,h)"""
    #     bsz, c, w, h = input_tensor.shape
    #     input_tensor = input_tensor.reshape(-1,input_tensor.shape[2]*input_tensor.shape[3],input_tensor.shape[1])
    #     query_layer = self.query(input_tensor)
    #     key_layer = self.key(input_tensor)
    #     value_layer = self.value(input_tensor)
    #
    #     # Take the dot product between "query" and "key" to get the raw attention scores.
    #     attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    #     attention_scores = attention_scores / math.sqrt(self.hidden_size)
    #
    #     # Normalize the attention scores to probabilities.
    #     attention_probs = nn.Softmax(dim=-1)(attention_scores)
    #     # attention_probs = self.out_dropout(attention_probs)
    #
    #     hidden_states = torch.matmul(attention_probs, value_layer)
    #     hidden_states = self.LayerNorm(hidden_states + input_tensor)
    #     hidden_states = hidden_states.view(-1, self.hidden_size, w, h)
    #
    #     return hidden_states

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

# class SelfAttention(nn.Module):
#     def __init__(self, input_size):
#         super(SelfAttention, self).__init__()
#
#         self.query = nn.Linear(input_size, input_size)
#         self.key = nn.Linear(input_size, input_size)
#         self.value = nn.Linear(input_size, input_size)
#
#         # self.out_dropout = nn.Dropout(dropout_prob)
#         self.hidden_size = input_size
#         self.LayerNorm = LayerNorm(input_size, eps=1e-12)
#
#     def forward(self, input_tensor):
#         bsz, c, w, h = input_tensor.shape
#         input_tensor = input_tensor.reshape(-1,input_tensor.shape[2]*input_tensor.shape[3],input_tensor.shape[1])
#         query_layer = self.query(input_tensor)
#         key_layer = self.key(input_tensor)
#         value_layer = self.value(input_tensor)
#
#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.hidden_size)
#
#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         # attention_probs = self.out_dropout(attention_probs)
#
#         hidden_states = torch.matmul(attention_probs, value_layer)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         hidden_states = hidden_states.view(-1, self.hidden_size, w, h)
#
#         return hidden_states

# class ResNet50_Encoder_Proj(nn.Module):
#     """backbone + projection head"""
#     def __init__(self, pretrained,head='mlp', feat_dim=128):
#         super(ResNet50_Encoder_Proj, self).__init__()
#         model = resnet50(pretrained=pretrained)
#         self.drop = nn.Dropout(0.5)
#         self.encoder = model
#         if head == 'linear':
#             self.fc = nn.Linear(2048, feat_dim)
#         elif head == 'mlp':
#             self.fc = nn.Sequential(
#                 nn.Linear(2048, 2048),
#                 self.drop,
#                 nn.ReLU(inplace=True),
#                 nn.Linear(2048, feat_dim)
#             )
#         else:
#             raise NotImplementedError(
#                 'head not supported: {}'.format(head))
#
#     def forward(self, x):
#         heat_map1, heat_map2, heat_map3, heat_map4, feat, out = self.encoder(x)# dim = 2048
#         feat = F.normalize(self.fc(feat), dim=1)#norm feature
#         return heat_map1, heat_map2, heat_map3, heat_map4, feat, out

# from fvcore.nn import FlopCountAnalysis
# net = resnet50(pretrained=False)
# net.fc = nn.Linear(2048, 100)
# net.fc_local = nn.Linear(64, 100)
# grid_mask = torch.randint(10, size=(32,32))
# tensor = (torch.randn(1, 3, 32, 32),True, grid_mask,0.7) # 如果有多个参数就写成tuple形式
#
#
# flops = FlopCountAnalysis(net, tensor)
# print("FLOPs:", flops.total()/1e9)
#
# papa_total = sum([param.nelement() for param in net.parameters()])
# print("Param", papa_total/1e6)


"""inference time"""
net = resnet50(pretrained=True)
net.fc = nn.Linear(2048, 100)
net = net.cuda()

iterations = 300
random_input = torch.randn(1,3,32,32).cuda()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

#GPU预热
for _ in range(50):
    _ = net(random_input)

#测速
times = torch.zeros(iterations)
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = net(random_input)
        ender.record()

        #同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) #计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time:{:.6f}, FPS: {}".format(mean_time, 1000/mean_time))