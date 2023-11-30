
import copy

import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torchvision import datasets
from networks.A0908_ResNet_CCAMix_model2_lessChannel import resnet50
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import datetime
import numpy as np
from scipy import stats
from skimage import segmentation
from losses.supcon_pixelwise import SupConLoss
import torch.nn.functional as F
import random


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
    sys.stdout = Logger(
        fileName + 'A0815_C100_32_3k3R50_CCAMix_SCL_Response07_3loss_LessChannel.log', path=path)


make_print_to_file(path='./logs')

# super param
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(
    description="CCAMix_self_C100_32_3k3R50")
parser.add_argument("--NUM_CLASS", type=int, default=100)
parser.add_argument("--data_path", type=str, default='./datasets/cifar100')
parser.add_argument("--RESUME", type=bool, default=True)
parser.add_argument("--START_EPOCH", type=int, default=500)
parser.add_argument("--NB_EPOCH", type=int, default=600)
parser.add_argument("--LR", type=float, default=0.02)
parser.add_argument("--BATCH_SIZE", "-b", type=int, default=32)
parser.add_argument("--gpu_ids", type=list, default=[0])
parser.add_argument("--p_binom", type=float, default=0.5)
parser.add_argument("--topN_local_ratio", type=float, default=0.7)
parser.add_argument("--loss_gamma_CE", type=float, default=0.1)
parser.add_argument("--loss_gamma_SCL", type=float, default=0.05)
parser.add_argument("--temp", type=float, default=0.7)
parser.add_argument("--N_min",  type=float, default=30)
parser.add_argument("--N_max",  type=float, default=35)
parser.add_argument("--cutmix_prob", type=float, default=0.5)
args = parser.parse_args()
print(args)


# normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

def cifar100_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # normalize
    ])

    cifar100_training = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size=args.BATCH_SIZE, shuffle=True,
                                              num_workers=8)

    cifar100_testing = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=128, shuffle=False, num_workers=8)

    print("number of CIFAR100 samples for training: ", len(cifar100_training))
    print("number of CIFAR100 samples for testing: ", len(cifar100_testing))

    return trainloader, testloader, len(cifar100_training), len(cifar100_testing)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def rand_bnl(p_binom, size):
    brl = stats.bernoulli.rvs(p_binom, size=size, random_state=None)  # random_state=None指每次生成随机
    # print(brl)
    (zero_idx,) = np.where(brl == int(0))
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
        SuperP_map_b = segmentation.slic(img_seg_b.cpu().numpy(), n_segments=N_b, compactness=8)

        SuperP_map_b = SuperP_map_b + 10000  # laverage to have differnet label value with SuperP_map_a
        SuperP_map_b_value = np.unique(SuperP_map_b)

        img_seg_a = image_a[sp].reshape(W, H, -1)
        N_a = random.randint(N_min, N_max)
        SuperP_map_a = segmentation.slic(img_seg_a.cpu().numpy(), n_segments=N_a, compactness=8)

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


name = './Models_out/A0815_C100_32_3k3R50_CCAMix_SCL_Response07_3loss_LessChannel_best.pt'
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
trainloader, testloader, len_train, len_test = cifar100_dataset()

models_out_dir = './Models_out/'
results_out_dir = './Results_out/'
if not os.path.exists(models_out_dir):
    os.makedirs(models_out_dir)
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

criterion_ce = nn.CrossEntropyLoss(reduction='none')
criterion_SCL = SupConLoss(temperature=args.temp)

Acc_best = 0.8308

tm = datetime.datetime.now().strftime('T' + '%m%d%H%M')
results_file_name = results_out_dir + tm + 'A0815_C100_32_3k3R50_CCAMix_SCL_Response07_3loss_LessChannel_results.txt'
with open(results_file_name, 'w') as file:
    file.write(
        'Epoch , train_acc , train_loss , train_loss_local_CE, train_loss_local_SCL, train_loss_global, test_acc , test_loss \n')
#############################################################################################
for epoch in range(args.START_EPOCH, args.NB_EPOCH):
    if epoch <= 9:
        if args.LR >= 0.00001:
            warmup_lr = args.LR * ((epoch+1) / 10)
            lr = warmup_lr
        else:
            lr = args.LR
    elif 9 < epoch <= 159:  # 40->80
        lr = args.LR
    elif 159 < epoch <= 199:
        lr = args.LR / 2
    elif 199 < epoch <= 249:
        lr = args.LR / 4
    elif 249 < epoch <= 299:
        lr = args.LR / 8
    elif 299 < epoch <= 349:  # 50
        lr = args.LR / 10
    elif 349 < epoch <= 399:
        lr = args.LR / 20
    elif 399 < epoch <= 449:
        lr = args.LR / 50
    elif 449 < epoch <= 499:  # 30
        lr = args.LR / 100
    elif 499 < epoch <= 549:  # 20
        lr = args.LR / 500
    else:
        lr = args.LR / 1000
    print("current lr:", lr)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_loss = 0
    train_loss_global = []
    train_loss_local_CE = []
    train_loss_local_SCL = []
    test_loss = 0
    total_correct_tr = 0
    total_correct_ts = 0

    for batch in tqdm(trainloader):
        images, labels = batch
        # print(images)
        images = images.float().cuda()
        labels = labels.long().cuda()
        net.train()

        r = np.random.rand(1)  # 从0-1正太分布数据中返回一个数
        if args.p_binom > 0 and r < args.cutmix_prob:
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = labels  # batch size个样本标签
            target_b = labels[rand_index]  # 将原有的batch打乱顺序

            img_mix, SuperP_map_batch_list, sel_idx_batch_list, nb_pixels_ALLsuperP_batch = \
                SuperpixelMixer(images,images[rand_index],args.N_min,args.N_max,args.p_binom)
            preds, preds_cell_list, weights_local, topN_idx = net(img_mix, local=True,
                                                                  superpixel_map=SuperP_map_batch_list,
                                                                  topN_local_ratio=args.topN_local_ratio)
            # print(preds_cell.shape) #torch.Size([32, 16, 100])

            lam_batch = []
            for sp in range(images.shape[0]):
                # print(weights_local[sp].shape, nb_pixels_ALLsuperP_batch[sp].shape)
                weighted_locals = weights_local[sp] * nb_pixels_ALLsuperP_batch[sp].cuda()
                # print(weighted_locals.shape, sel_idx_batch_list[sp])
                sel_weights_locals = [weighted_locals[s] for s in sel_idx_batch_list[sp]]
                nb_pixel_locals = [nb_pixels_ALLsuperP_batch[sp][s] for s in sel_idx_batch_list[sp]]

                if len(sel_weights_locals) > 0:
                    lam1 = torch.stack(sel_weights_locals).sum() / weighted_locals.sum()
                    lam0 = torch.stack(nb_pixel_locals).sum() / nb_pixels_ALLsuperP_batch[sp].sum()
                else:
                    lam0, lam1 = 0, 0
                lam = (lam0 + lam1) / 2
                # print(lam)
                if lam > 1:
                    print("ERROR: lamda over one")
                lam_batch.append(lam)
            lam_batch = torch.as_tensor(lam_batch).cuda()
            loss_total = (
                        torch.mul(criterion_ce(preds, target_a), (1. - lam_batch)) + torch.mul(criterion_ce(preds, target_b),
                                                                                            lam_batch)).mean()
            # print(loss_total)

            loss_local_ce = 0
            fea_SCL_batch = preds_cell_list[0]
            label_SCL_batch_tp = target_a[0].repeat(np.unique(SuperP_map_batch_list[0]).shape[0])
            for idx in range(np.unique(SuperP_map_batch_list[0]).shape[0]):
                if np.unique(SuperP_map_batch_list[0])[idx] > 9999:
                    label_SCL_batch_tp[idx] = target_b[0]
            if len(topN_idx[0]) > 0:
                label_SCL_batch_tp = [label_SCL_batch_tp[idd] for idd in topN_idx[0]]
                label_SCL_batch_tp = torch.stack(label_SCL_batch_tp)
            label_SCL_batch = label_SCL_batch_tp
            # print(fea_SCL_batch.shape[0], label_SCL_batch.shape[0])

            for sp in range(0, images.shape[0]):
                target_cells_current = target_a[sp].repeat(np.unique(SuperP_map_batch_list[sp]).shape[0])
                for idx in range(np.unique(SuperP_map_batch_list[sp]).shape[0]):
                    if np.unique(SuperP_map_batch_list[sp])[idx] > 9999:
                        target_cells_current[idx] = target_b[sp]
                if len(topN_idx[sp]) > 0:
                    target_cells_current = [target_cells_current[idd] for idd in topN_idx[sp]]
                    target_cells_current = torch.stack(target_cells_current)

                loss_cell = criterion_ce(preds_cell_list[sp], target_cells_current).sum()
                # print(loss_cell)
                loss_local_ce = loss_local_ce + loss_cell

                if sp < (images.shape[0] -1):
                    # print(preds_cell_list[sp].shape[0])
                    target_cells_next = target_a[sp+1].repeat(np.unique(SuperP_map_batch_list[sp+1]).shape[0])
                    for idx in range(np.unique(SuperP_map_batch_list[sp+1]).shape[0]):
                        if np.unique(SuperP_map_batch_list[sp+1])[idx] > 9999:
                            target_cells_next[idx] = target_b[sp+1]
                    if len(topN_idx[sp+1]) > 0:
                        target_cells_next = [target_cells_next[idd] for idd in topN_idx[sp+1]]
                        target_cells_next = torch.stack(target_cells_next)
                    fea_SCL_batch = torch.cat((fea_SCL_batch, preds_cell_list[sp+1]), dim=0)
                    label_SCL_batch = torch.cat((label_SCL_batch, target_cells_next), dim=0)
                   # print(fea_SCL_batch.shape[0], label_SCL_batch.shape[0])

            # print(fea_SCL_batch.unsqueeze(1).shape)
            # fea_SCL_batch = fea_SCL_batch.unsqueeze(1).expand(1,100)
            # print(fea_SCL_batch.shape)

            loss_local_SCL = criterion_SCL(F.normalize(fea_SCL_batch,dim=1), label_SCL_batch)
            loss = args.loss_gamma_CE * loss_local_ce / images.shape[0] + loss_total + args.loss_gamma_SCL * loss_local_SCL
            train_loss_global.append(loss_total)
            train_loss_local_CE.append(args.loss_gamma_CE * loss_local_ce / images.shape[0])
            train_loss_local_SCL.append(args.loss_gamma_SCL * loss_local_SCL)
            # print("local_CE_loss:", args.loss_gamma_CE * loss_local_ce / images.shape[0])
            # print("local_SCL_loss:", args.loss_gamma_SCL * loss_local_SCL)
            # print("total_CE_loss:", loss_total)
        else:
            # compute output
            preds = net(images)
            loss = criterion_ce(preds, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_correct_tr += get_num_correct(preds, labels)

        del images;
        del labels;


