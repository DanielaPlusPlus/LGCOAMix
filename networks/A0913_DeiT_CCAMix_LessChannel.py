"""
https://github.com/facebookresearch/deit/blob/main/README_deit.md?plain=1
https://github.com/facebookresearch/deit/blob/main/models.py
"""
# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

# import torch
# # check you have the right version of timm
# import timm
# assert timm.__version__ == "0.3.2"
#
# # now load it with torchhub
# DeiT = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import math
from typing import Dict, Iterable, Callable
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        # print(x.shape)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        # print(x.shape)

        x = self.norm(x)
        # print(x.shape)  #torch.Size([2, 198, 768])
        # print(x[:, 0].shape) #torch.Size([2, 768])
        # print(x[:, 1].shape) #torch.Size([2, 768])
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model




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


"""
https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
"""
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


class DeiTB_CCAMix(nn.Module):
    def __init__(self, pretrained, NUM_CLASS):
        super().__init__()
        # net_name = './Models_out/CUB_224_CE_DeiTB_OcCaMix_best.pt'
        # if len(args.gpu_ids) > 1:
        #     if args.RESUME:
        #         net = deit_base_patch16_224(pretrained=pretrained)
        #         net.head = nn.Linear(768, NUM_CLASS)
        #         net.load_state_dict(torch.load(net_name))
        #         # net = net.cuda()
        #     else:
        #         net = deit_base_patch16_224(pretrained=True)
        #         net.head = nn.Linear(768, NUM_CLASS)
        #         # net = net.cuda()
        #     net = torch.nn.DataParallel(net)
        # else:
        #     if args.RESUME:
        #         net = deit_base_patch16_224(pretrained=True)
        #         net.head = nn.Linear(768, NUM_CLASS)
        #         net.load_state_dict(torch.load(net_name))
        #         # net = net.cuda()
        #     else:
        #         net = deit_base_patch16_224(pretrained=True)
        #         net.head = nn.Linear(768, NUM_CLASS)
        #         # net = net.cuda()
        net = deit_base_patch16_224(pretrained=pretrained)
        self.encoder = FeatureExtractor(net, layers=['norm', "head"])

        #adding
        # self.relu = nn.ReLU(inplace=True)
        # self.conv5 = conv(768, 512, stride=2, transposed=True)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.conv6 = conv(512, 256, stride=2, transposed=True)
        # self.bn6 = nn.BatchNorm2d(256)
        # self.conv7 = conv(256, 128, stride=2, transposed=True)
        # self.bn7 = nn.BatchNorm2d(128)
        # self.conv8 = conv(128, 64, stride=2, transposed=True)
        # self.bn8 = nn.BatchNorm2d(64)
        #
        # self.SA = SelfAttention(input_size=64)
        # self.SAP = self.SuperpixelAttentionPooling
        # self.fc_local = nn.Linear(64, NUM_CLASS)


        self.relu = nn.ReLU(inplace=True)
        self.conv5 = conv(768, 256, stride=2, transposed=True)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True)
        self.bn8 = nn.BatchNorm2d(64)

        self.SA = SelfAttention(input_size=64)
        self.SAP = self.SuperpixelAttentionPooling
        self.fc_local = nn.Linear(64, NUM_CLASS)

    def forward(self, x, local=False, superpixel_map=None, topN_local_ratio=None):
        features = self.encoder(x)
        fea = features["norm"][:, 0:196, :].reshape(features["norm"][:, 0:196, :].shape[0], features["norm"][:, 0:196, :].shape[2], 14, 14)
        logits = features["head"]

        """local"""
        if local:
            # fea [8,768,14,14]
            x_up1 = self.relu(self.bn5(self.conv5(fea)))
            # print(x_up1.shape)  #torch.Size([8, 512, 28, 28])
            x_up2 = self.relu(self.bn6(self.conv6(x_up1)))
            # print(x_up2.shape)  #torch.Size([8, 256, 56, 56])
            x_up3 = self.relu(self.bn7(self.conv7(x_up2)))
            # print(x_up3.shape)  #torch.Size([8, 128, 112, 112])
            x_up4 = self.bn8(self.conv8(x_up3)) #(8,64,224,224)

            SPAttention_batch, topN_SurPFeat_batch, topN_SurPFeat_idx_batch = self.SAP(x_up4, superpixel_map, topN_local_ratio)  # list(32*[n, 256])

            topN_SurPLocalCls_batch = []
            for sp in range(len(topN_SurPFeat_batch)):
                x_locals_out_sp = []
                for tk in range(topN_SurPFeat_batch[sp].shape[0]):
                    x_local_out = self.fc_local(topN_SurPFeat_batch[sp][tk])
                    x_locals_out_sp.append(x_local_out)
                x_locals_out_sp = torch.stack(x_locals_out_sp)
                topN_SurPLocalCls_batch.append(x_locals_out_sp)
            """output: global_preds; selected local preds; attention for every locals; selected local index; final feature"""
            return logits, topN_SurPLocalCls_batch, topN_SurPFeat_idx_batch, SPAttention_batch,  x_up4
        else:
            return logits

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

