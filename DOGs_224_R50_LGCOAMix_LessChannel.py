#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from networks.A0911_CCAMix_ResNet_CUB224_LessChannel import resnet50
from dataset_StanfordDogs import Dogs
import torchvision.transforms as transforms
import copy
from tqdm import tqdm
import os
import numpy as np
import datetime
import random
from losses.supcon_pixelwise_RandomSelection import SupConLoss
from scipy import stats
from skimage import segmentation
import sys
import torch.nn.functional as F

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('DAY' + '%Y_%m_%d_')
    sys.stdout = Logger(fileName + 'DOGs_224_R50_CCAMix_LessChannel.log', path=path)


make_print_to_file(path='./logs')

# super param
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="DOGs_224_R50_CCAMix")
parser.add_argument("--NUM_CLASS", type=int, default=120)
parser.add_argument("--RESUME", type=bool, default=True)
parser.add_argument("--START_EPOCH", type=int, default=470)
parser.add_argument("--NB_EPOCH", type=int, default=600)
parser.add_argument("--LR", type=float, default=0.01)
parser.add_argument("--BATCH_SIZE", "-b", type=int, default=16)
parser.add_argument("--gpu_ids", type=list, default=[0])
parser.add_argument("--Burn_prob", type=float, default=0.5)
parser.add_argument("--topN_local_ratio", type=float, default=0.7)
parser.add_argument("--loss_gamma_CE", type=float, default=0.03)
parser.add_argument("--loss_gamma_SCL", type=float, default=0.05)
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--N_min",  type=float, default=40)
parser.add_argument("--N_max",  type=float, default=60)
parser.add_argument("--cutmix_prob", type=float, default=0.5)

args = parser.parse_args()
print(args)

train_transform = transforms.Compose([
    # transforms.ToPILImage(mode='RGB'),
    transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])


test_transform = transforms.Compose([
    # transforms.ToPILImage(mode='RGB'),
    transforms.Resize([224, 224]),
    # transforms.CenterCrop([448, 448]),
    transforms.ToTensor(),
])

train_set = Dogs(
    root = './datasets/dogs',
    train = True,
    transform = train_transform,
    download = False
)

test_set = Dogs(
    root = './datasets/dogs',
    train = False,
    transform=test_transform,
    download = False
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size =args.BATCH_SIZE, shuffle = True)
test_loader  = torch.utils.data.DataLoader(test_set, batch_size = 128)
print('Number of Samples in training set：',len(train_set))
print('Number of Samples in test set：',len(test_set))

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def rand_bnl(size, p_binom=0.5):
    brl = stats.bernoulli.rvs(p_binom, size=size, random_state=None)  # random_state=None指每次生成随机
    # print(brl)
    (zero_idx,) = np.where(brl == int(1))
    return zero_idx


def SuperpixelMixer(image_a, image_b, N_min, N_max, Bern_prob):
    bsz, C, W, H = image_a.shape
    binary_mask = []
    SuperP_map_batch_list = []
    sel_idx_batch_list = []
    nb_pixels_ALLsuperP_batch = []
    for sp in range(bsz):
        img_seg_b = image_b[sp].reshape(W, H, -1)  # (W,H,C)
        N_b = random.randint(N_min, N_max)
        SuperP_map_b = segmentation.slic(img_seg_b.cpu().numpy(), n_segments=N_b, compactness=10)

        SuperP_map_b = SuperP_map_b + 10000  # laverage to have differnet label value with SuperP_map_a
        SuperP_map_b_value = np.unique(SuperP_map_b)

        img_seg_a = image_a[sp].reshape(W, H, -1)
        N_a = random.randint(N_min, N_max)
        SuperP_map_a = segmentation.slic(img_seg_a.cpu().numpy(), n_segments=N_a, compactness=10)

        sel_region_idx = rand_bnl(p_binom=Bern_prob, size=SuperP_map_b_value.shape[0])

        binary_mask_sp_b = np.zeros((W, H))
        for v in range(SuperP_map_b_value.shape[0]):
            if v in sel_region_idx:
                bool_v = (SuperP_map_b == SuperP_map_b_value[v])
                binary_mask_sp_b[bool_v == True] = 1  # mix处mask是1, 否则是0
            else:
                pass

        SuperP_map = SuperP_map_a * (1 - binary_mask_sp_b) + SuperP_map_b * binary_mask_sp_b

        """attention calculation on final superpixel map"""
        SuperP_map_value = np.unique(SuperP_map)
        nb_pixels_ALLsuperP = []
        idx_final_map = []
        for v in range(SuperP_map_value.shape[0]):
            bool_v_all = (SuperP_map == SuperP_map_value[v])
            nb_pixels_ALLsuperP.append(len(SuperP_map[bool_v_all == True]))
            if SuperP_map_value[v] > 9999:
                idx_final_map.append(v)

        binary_mask_sp_b = torch.tensor(binary_mask_sp_b).cuda()
        binary_mask_ch_sp_b = copy.deepcopy(binary_mask_sp_b)
        binary_mask_ch_sp_b = binary_mask_ch_sp_b.expand(C, -1, -1)  # torch.Size([3, 32, 32])
        binary_mask.append(binary_mask_ch_sp_b)

        SuperP_map_batch_list.append(torch.tensor(SuperP_map))
        sel_idx_batch_list.append(idx_final_map)
        nb_pixels_ALLsuperP_batch.append(torch.as_tensor(nb_pixels_ALLsuperP))

    binary_mask = torch.stack(binary_mask)
    binary_mask = binary_mask.float()
    img_mix = image_a * (1 - binary_mask) + image_b * binary_mask

    return img_mix, SuperP_map_batch_list, sel_idx_batch_list, nb_pixels_ALLsuperP_batch


name = './Models_out/DOGs_224_R50_CCAMix_LessChannel_best.pt'
if len(args.gpu_ids) > 1:
    if args.RESUME:
        net = resnet50(pretrained=False)
        net.fc = nn.Linear(2048, args.NUM_CLASS)
        net.fc_local = nn.Linear(64, args.NUM_CLASS)
        net.load_state_dict(torch.load(name))
        net = net.cuda()
    else:
        net = resnet50(pretrained=False)
        net.fc = nn.Linear(2048, args.NUM_CLASS)
        net.fc_local = nn.Linear(64, args.NUM_CLASS)
        net = net.cuda()
    encoder = torch.nn.DataParallel(net)
else:
    if args.RESUME:
        net = resnet50(pretrained=False)
        net.fc = nn.Linear(2048, args.NUM_CLASS)
        net.fc_local = nn.Linear(64, args.NUM_CLASS)
        net.load_state_dict(torch.load(name))
        net = net.cuda()
    else:
        net = resnet50(pretrained=False)
        net.fc = nn.Linear(2048, args.NUM_CLASS)
        net.fc_local = nn.Linear(64, args.NUM_CLASS)
        net = net.cuda()

models_out_dir = './Models_out/'
results_out_dir = './Results_out/'
if not os.path.exists(models_out_dir):
    os.makedirs(models_out_dir)
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

criterion_ce = nn.CrossEntropyLoss(reduction='none')
criterion_SCL = SupConLoss(temperature=args.temp)

Acc_best = 0.7039

tm = datetime.datetime.now().strftime('T' + '%m%d%H%M')
results_file_name = results_out_dir +  tm + 'DOGs_224_R50_CCAMix_LessChannel_results.txt'
with open(results_file_name, 'w') as file:
    file.write(
        'Epoch , train_acc , train_loss , test_acc , test_loss \n')
#############################################################################################
for epoch in range(args.START_EPOCH, args.NB_EPOCH):
    if epoch <= 9:
        if args.LR >= 0.00001:
            warmup_lr = args.LR * ((epoch+1) / 10)
            lr = warmup_lr
        else:
            lr = args.LR
    elif 9 < epoch <= 129:  # 130
        lr = args.LR
    elif 129 < epoch <= 169: # 40
        lr = args.LR / 2
    elif 169 < epoch <= 219: # 50
        lr = args.LR / 4
    elif 219 < epoch <= 259: # 40
        lr = args.LR / 8
    elif 259 < epoch <= 309:  # 50
        lr = args.LR / 10
    elif 309 < epoch <= 349:  # 40
        lr = args.LR / 20
    elif 349 < epoch <= 389:  # 40
        lr = args.LR / 40
    elif 389 < epoch <= 419:  # 30
        lr = args.LR / 80
    elif 419 < epoch <= 469:  # 50
        lr = args.LR / 100
    elif 469 < epoch <= 499:  # 30
        lr = args.LR / 200
    elif 499 < epoch <= 529:  # 30
        lr = args.LR / 400
    elif 529 < epoch <= 559:  # 30
        lr = args.LR / 800
    elif 559 < epoch <= 579:  # 20
        lr = args.LR / 1000
    else:
        lr = args.LR / 5000  # 20
    print("current lr:", lr)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_loss = 0
    train_loss_global = []
    train_loss_local_CE = []
    train_loss_local_SCL = []
    test_loss = 0
    total_correct_tr = 0
    total_correct_ts = 0

    for batch in tqdm(train_loader):
        images, labels = batch
        images = images.float().cuda()
        labels = labels.long().cuda()
        # bsz,C,W,H = images.shape

        r = np.random.rand(1)  # 从0-1正太分布数据中返回一个数
        if args.Burn_prob > 0 and r < args.cutmix_prob:
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = labels  # batch size个样本标签
            target_b = labels[rand_index]  # 将原有的batch打乱顺序

            img_mix, SuperP_map_batch_list, sel_idx_batch_list, nb_pixels_ALLsuperP_batch = \
                SuperpixelMixer(images, images[rand_index], args.N_min, args.N_max, args.Burn_prob)

            net.train()
            preds, topN_preds_local, topN_idx_local, weights_local, feat_pixel = net(img_mix, local=True, superpixel_map=SuperP_map_batch_list,
                                                                  topN_local_ratio=args.topN_local_ratio)
            # print(weights_local.shape) #torch.Size([32, 16, 100])


            """----------------global classification------------------"""
            #----------------------pixel-based for lam
            # attention_pixel = F.sigmoid(feat_pixel.sum(1))
            # print(feat_pixel.shape, attention_pixel.shape) #torch.Size([32, 64, 32, 32]) torch.Size([32, 32, 32])
            # lam_batch = []
            # for sp in range(images.shape[0]):
            #     """lamda calculation """
            #     pixels_PerLocal = []
            #     nb_pixels_PerLocal = []
            #     idx_b_MixLocal = []
            #     SuperP_map_value = np.unique(SuperP_map_batch_list[sp])
            #
            #     for v in range(SuperP_map_value.shape[0]):
            #         bool_v_all = (SuperP_map_batch_list[sp] == SuperP_map_value[v])
            #         nb_pixels_PerLocal.append(len(SuperP_map_batch_list[sp][bool_v_all == True]))
            #         x_tuple, y_tuple = np.where(bool_v_all == True)
            #         xy = []
            #         for xx in range(0, len(x_tuple)):
            #             xy.append([x_tuple[xx], y_tuple[xx]])
            #         pixels_PerLocal.append(xy)
            #         # print(len(xy))
            #         if SuperP_map_value[v] > 9999:
            #             idx_b_MixLocal.append(v)  # the local index from source image(b) for the mix image
            #     nb_pixels_PerLocal = torch.tensor(nb_pixels_PerLocal).cuda()
            #
            #     pixels_b_PerLocal = []
            #     if len(idx_b_MixLocal) > 0:
            #         pixels_b_AllLocal = []
            #         for ss in idx_b_MixLocal:
            #             pixels_b_AllLocal.extend(pixels_PerLocal[ss])
            #         # if len(pixels_b_AllLocal) < attention_pixel[sp].shape[0] * attention_pixel[sp].shape[1]:
            #         attention_pixels_b_AllLocal = [attention_pixel[sp][s[0]][s[1]] for s in pixels_b_AllLocal]
            #         attention_pixels_b_AllLocal = torch.tensor(attention_pixels_b_AllLocal)
            #         AreaWeights_b_MixLocals = [len(pixels_PerLocal[s]) for s in idx_b_MixLocal]  # 计算出总mix region的area
            #         attention_pixel_flatten = attention_pixel[sp].reshape(
            #             attention_pixel[sp].shape[0] * attention_pixel[sp].shape[1])
            #         lam_atten = attention_pixels_b_AllLocal.sum() / attention_pixel_flatten.sum()
            #         lam_area = torch.tensor(AreaWeights_b_MixLocals).sum() / nb_pixels_PerLocal.sum()
            #         if lam_atten > 1:
            #             lam_atten = 1
            #             # print(len(pixels_b_AllLocal), float(attention_pixels_b_AllLocal.sum()),
            #             #       float(attention_pixel_flatten.sum()), float(lam_atten))
            #             # print(attention_pixels_b_AllLocal)
            #             # print(attention_pixel_flatten)
            #     else:
            #         lam_atten, lam_area = 0, 0
            #     lam = lam_atten
            #     if lam > 1:
            #         print("ERROR: lamda over one")
            #     lam_batch.append(lam)
            # lam_batch = torch.as_tensor(lam_batch).cuda()
            # loss_total = (
            #         torch.mul(criterion_ce(preds, target_a), (1. - lam_batch)) + torch.mul(
            #     criterion_ce(preds, target_b),
            #     lam_batch)).mean()


            #----------------------SP-based for lam
            lam_batch = []
            for sp in range(images.shape[0]):
                # print(weights_local[sp].shape, nb_pixels_ALLsuperP_batch[sp].shape)
                weighted_locals = weights_local[sp] * nb_pixels_ALLsuperP_batch[sp].cuda()
                # print(weighted_locals.shape, sel_idx_batch_list[sp])
                sel_weights_locals = [weighted_locals[s] for s in sel_idx_batch_list[sp]]
                nb_pixel_locals = [nb_pixels_ALLsuperP_batch[sp][s] for s in sel_idx_batch_list[sp]]  # mix的像素数量

                if len(sel_weights_locals) > 0:
                    lam = torch.stack(sel_weights_locals).sum() / weighted_locals.sum() # SP-based
                    # lam0 = torch.stack(nb_pixel_locals).sum() / nb_pixels_ALLsuperP_batch[sp].sum()
                else:
                    # lam0, lam1 = 0, 0
                    lam = 0
                # print(lam)
                if lam > 1:
                    lam = 1
                    print("ERROR: lamda over one")
                lam_batch.append(lam)
            lam_batch = torch.as_tensor(lam_batch).cuda()
            loss_total = (
                    torch.mul(criterion_ce(preds, target_a), (1. - lam_batch)) + torch.mul(
                criterion_ce(preds, target_b),
                lam_batch)).mean()

            """----------------contrastive learning and local classification------------------"""
            """inital 1st-batch feature and label for batch-based SCL loss calculation"""
            loss_local_ce = 0
            fea_SCL_batch = topN_preds_local[0]
            label_SCL_batch_tp = target_a[0].repeat(np.unique(SuperP_map_batch_list[0]).shape[0])
            for idx in range(np.unique(SuperP_map_batch_list[0]).shape[0]):
                if np.unique(SuperP_map_batch_list[0])[idx] > 9999:
                    label_SCL_batch_tp[idx] = target_b[0]
            label_SCL_batch_tp = [label_SCL_batch_tp[idd] for idd in topN_idx_local[0]]
            label_SCL_batch_tp = torch.stack(label_SCL_batch_tp)
            label_SCL_batch = label_SCL_batch_tp
            # print(fea_SCL_batch.shape[0], label_SCL_batch.shape[0])

            """current-batch feature and label for local classification;  """
            loss_local_ce = 0
            for sp in range(0, images.shape[0]):
                target_cells_current = target_a[sp].repeat(np.unique(SuperP_map_batch_list[sp]).shape[0])
                for idx in range(np.unique(SuperP_map_batch_list[sp]).shape[0]):
                    if np.unique(SuperP_map_batch_list[sp])[idx] > 9999:
                        target_cells_current[idx] = target_b[sp]
                target_cells_current = [target_cells_current[idd] for idd in topN_idx_local[sp]]
                target_cells_current = torch.stack(target_cells_current)

                loss_cell = criterion_ce(topN_preds_local[sp], target_cells_current).sum()
                loss_local_ce = loss_local_ce + loss_cell

                """next-batch feature and label for batch-based SCL loss calculation and"""
                if sp < (images.shape[0] -1):
                    # print(preds_cell_list[sp].shape[0])
                    target_cells_next = target_a[sp+1].repeat(np.unique(SuperP_map_batch_list[sp+1]).shape[0])
                    for idx in range(np.unique(SuperP_map_batch_list[sp+1]).shape[0]):
                        if np.unique(SuperP_map_batch_list[sp+1])[idx] > 9999:
                            target_cells_next[idx] = target_b[sp+1]
                    target_cells_next = [target_cells_next[idd] for idd in topN_idx_local[sp+1]]
                    target_cells_next = torch.stack(target_cells_next)

                    fea_SCL_batch = torch.cat((fea_SCL_batch, topN_preds_local[sp+1]), dim=0)
                    label_SCL_batch = torch.cat((label_SCL_batch, target_cells_next), dim=0)
                   # print(fea_SCL_batch.shape[0], label_SCL_batch.shape[0])

            # print(fea_SCL_batch.unsqueeze(1).shape)
            # fea_SCL_batch = fea_SCL_batch.unsqueeze(1).expand(1,100)
            # print(fea_SCL_batch.shape)

            # print("fea_SCL_batch:", F.normalize(fea_SCL_batch,dim=1))
            # print("label_SCL_batch:", label_SCL_batch)

            loss_local_SCL = criterion_SCL(F.normalize(fea_SCL_batch,dim=1), label_SCL_batch)
            # print(label_SCL_batch)
            loss = args.loss_gamma_CE * loss_local_ce / images.shape[0] + loss_total + args.loss_gamma_SCL * loss_local_SCL
            # loss =args.loss_gamma_CE * loss_local_ce / images.shape[0] + loss_total
            train_loss_global.append(loss_total)
            train_loss_local_CE.append(args.loss_gamma_CE * loss_local_ce / images.shape[0])
            train_loss_local_SCL.append(args.loss_gamma_SCL * loss_local_SCL)
            # print("local_CE_loss:", args.loss_gamma_CE * loss_local_ce / images.shape[0])
            # print("local_SCL_loss:", args.loss_gamma_SCL * loss_local_SCL)
            # print("total_CE_loss:", loss_total)

            if np.any(np.isnan(loss_local_ce.cpu().detach().numpy())):
                print("local_CE_loss NAN:",loss_local_ce)
                # print(topN_preds_local)
                sys.exit()
            if np.any(np.isnan(loss_total.cpu().detach().numpy())):
                print("total_CE_loss NAN:",loss_total)
                # print(preds)
                # print(lam_batch)
                # print(target_a, target_b)
                sys.exit()
            if np.any(np.isnan(loss_local_SCL.cpu().detach().numpy())):
                print("local_SCL_loss NAN:",loss_local_SCL)
                print(F.normalize(fea_SCL_batch,dim=1))
                print(label_SCL_batch)
                sys.exit()
        else:
            # compute output
            net.train()
            preds = net(images)
            loss = criterion_ce(preds, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_correct_tr += get_num_correct(preds, labels)

        del images;
        del labels;

