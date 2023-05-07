"""
https://github.com/tfzhou/ContrastiveSeg/blob/main/lib/loss/loss_contrast.py
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, feats_, labels_):
        # anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        anchor_num, n_view = feats_.shape[0], 2  #anchor_num=输入的向量个数，n_view=正负两种views

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()#计算label对应的equal矩阵

        contrast_count = n_view
        # contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        contrast_feature = feats_

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)), #计算特征之间的点乘距离，结合温度系数
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() #减去最大值让数值更加稳定
        # print("logits_max.detach():",logits_max.detach())

        # mask = mask.repeat(anchor_count, contrast_count) #两倍复制（前提是label对应不同view的label相同）
        neg_mask = 1 - mask

        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
        #                                              0)
        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(anchor_num).view(-1, 1).cuda(),
        #                                              0)
        # mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #公式分母第二部分

        exp_logits = torch.exp(logits) #公式分子 = 公式分母第一部分

        log_prob = logits - torch.log(exp_logits + neg_logits) #分子处以分母后取log
        # print("exp_logits + neg_logits:", exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print(mask.sum(1))

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    # def forward(self, feats_, labels_):
    #     anchor_num, n_view = feats_.shape[0], feats_.shape[1]
    #
    #     labels_ = labels_.contiguous().view(-1, 1)
    #     mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
    #
    #     contrast_count = n_view
    #     contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
    #
    #     anchor_feature = contrast_feature
    #     anchor_count = contrast_count
    #
    #     anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
    #                                     self.temperature)
    #     logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    #     logits = anchor_dot_contrast - logits_max.detach()
    #
    #     mask = mask.repeat(anchor_count, contrast_count)
    #     neg_mask = 1 - mask
    #
    #     logits_mask = torch.ones_like(mask).scatter_(1,
    #                                                  torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
    #                                                  0)
    #     mask = mask * logits_mask
    #
    #     neg_logits = torch.exp(logits) * neg_mask
    #     neg_logits = neg_logits.sum(1, keepdim=True)
    #
    #     exp_logits = torch.exp(logits)
    #
    #     log_prob = logits - torch.log(exp_logits + neg_logits)
    #
    #     mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    #
    #     loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    #     loss = loss.mean()
    #
    #     return loss
