import torchvision
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch
import numpy as np
import itertools
import torch.nn.functional as F

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

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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
        # 平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#修改点3
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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

    def forward(self, x):
        # Perform the usual forward pass
        x = self.conv1_3k3(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x1 = self.layer1_3k3(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x_pool = self.avg_pool(x4)
        # print(x.shape)
        x_fea = x_pool.view(x_pool.size(0), -1)
        # print(x_fea.shape)
        x_fc = self.fc(x_fea)
        # print(x_fc.shape)

        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # torch.Size([256, 256, 32, 32])
        # torch.Size([256, 512, 16, 16])
        # torch.Size([256, 1024, 8, 8])
        # torch.Size([256, 2048, 4, 4])

        return x4,x_fc #(bs,2048,3,3),2048

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


from fvcore.nn import FlopCountAnalysis
net = resnet50(pretrained=True)
net.fc = nn.Linear(2048, 100)

tensor = (torch.randn(1, 3, 32, 32),) # 如果有多个参数就写成tuple形式


flops = FlopCountAnalysis(net, tensor)
print("FLOPs:", flops.total()/1e9)

papa_total = sum([param.nelement() for param in net.parameters()])
print("Param", papa_total/1e6)



# """inference time"""
# iterations = 300
# random_input = torch.randn(1,3,224,224).cuda()
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#
# #GPU预热
# for _ in range(50):
#     _ = net(random_input)
#
# #测速
# times = torch.zeros(iterations)
# with torch.no_grad():
#     for iter in range(iterations):
#         starter.record()
#         _ = net(random_input)
#         ender.record()
#
#         #同步GPU时间
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time() #计算时间
#         times[iter] = curr_time
#
# mean_time = times.mean().item()
# print("Inference time:{:.6f}, FPS: {}".format(mean_time, 1000/mean_time))
