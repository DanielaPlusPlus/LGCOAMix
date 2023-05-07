#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import random
import torch.nn as nn
import torch.optim
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
import argparse
import datetime
from networks.preResNet_CIFAR_1 import preactresnet18
from skimage import segmentation
import warnings
warnings.filterwarnings('ignore')

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

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
    sys.stdout = Logger(fileName + 'TinyImageNet_pre3k3R18_SuperpixelGridMix_64E8N.log', path=path)
make_print_to_file(path='./logs')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="TinyImageNet_pre3k3R18_SuperpixelGridMix_64E8N")
parser.add_argument("--NUM_CLASS", type=int, default=200)
parser.add_argument("--RESUME", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./datasets/tiny-imagenet-200')
parser.add_argument("--START_EPOCH", type=int, default=500)
parser.add_argument("--NB_EPOCH", type=int, default=600)
parser.add_argument("--LR", type=float, default=0.02)
parser.add_argument("--BATCH_SIZE", "-b", type=int, default=100)
parser.add_argument("--gpu_ids",  type=list, default=[0])
parser.add_argument("--cutmix_prob", type=float, default=0.5)
parser.add_argument("--ratio", type=float, default=0.4) #50
parser.add_argument("--SuperP_size", type=int, default=64)
parser.add_argument("--compact", type=float, default=7)
args = parser.parse_args()
print(args)

mean = [x / 255 for x in [127.5, 127.5, 127.5]]
std = [x / 255 for x in [127.5, 127.5, 127.5]]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])

train_root = os.path.join(args.data_path, 'train')  # this is path to training images folder
validation_root = os.path.join(args.data_path, 'val')  # this is path to validation images folder
train_set = datasets.ImageFolder(train_root, transform=train_transform)
test_set = datasets.ImageFolder(validation_root, transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                       batch_size=args.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=8)
test_loader = torch.utils.data.DataLoader(test_set,
                                   batch_size=256,
                                   shuffle=False,
                                   num_workers=8)
print('Number of Samples in training set：',len(train_set))
print('Number of Samples in test set：',len(test_set))


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def superpixel_loc(img, n_seg, compact):
    img = img.view(img.shape[1],img.shape[2],-1)#torch.Size([224, 224, 3])
    segments = segmentation.slic(img.cpu().numpy(), n_segments=n_seg,compactness=compact) #(224, 224)
    mask = np.unique(segments)#mask.shape=(n,)
    nb_SegLabel, = mask.shape
    # print(nb_SegLabel)
    x_idxs, y_idxs = [], []
    for i in range(0, nb_SegLabel):
        p = np.random.rand(1)
        if p < args.ratio:
            x, y = np.where(segments==mask[i])
            x_idxs.extend(x)
            y_idxs.extend(y)
    return x_idxs, y_idxs


net_name = './Models_out/TinyImageNet_pre3k3R18_SuperpixelGridMix_64E8N_best.pt'
if len(args.gpu_ids) > 1:
    if args.RESUME:
        net = preactresnet18(num_classes=args.NUM_CLASS)
        net.load_state_dict(torch.load(net_name))
        net = net.cuda()
    else:
        net = preactresnet18(num_classes=args.NUM_CLASS)
        net = net.cuda()
    model = torch.nn.DataParallel(net)
else:
    if args.RESUME:
        net = preactresnet18(num_classes=args.NUM_CLASS)
        net.load_state_dict(torch.load(net_name))
        net = net.cuda()
    else:
        net = preactresnet18(num_classes=args.NUM_CLASS)
        net = net.cuda()

models_out_dir = './Models_out/'
results_out_dir = './Results_out/'
if not os.path.exists(models_out_dir):
    os.makedirs(models_out_dir)
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

Acc_best = 0.6735
tm = datetime.datetime.now().strftime('T' + '%m%d%H%M')
results_file_name = results_out_dir +  tm + 'TinyImageNet_pre3k3R18_SuperpixelGridMix_64E8N_results.txt'
with open( results_file_name , 'w') as file:
    file.write(
        'Epoch , train_acc , train_loss , test_acc , test_loss \n')

criterion = nn.CrossEntropyLoss()

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
    val_loss = 0
    test_loss = 0
    total_correct_tr = 0
    total_correct_val = 0
    total_correct_ts = 0

    for batch in tqdm(train_loader):
        images, labels = batch
        images = images.float().cuda()
        labels = labels.long().cuda()
        bsz,C,W,H = images.shape
        net.train()
        r = np.random.rand(1) #从0-1正太分布数据中返回一个数
        if r < args.cutmix_prob:
            rand_index = torch.randperm(bsz).cuda()
            target_a = labels#batch size个样本标签
            target_b = labels[rand_index]#将原有的batch打乱顺序
            lam = []
            for k in range(0,bsz):
                with torch.no_grad():
                    x_idxs, y_idxs = superpixel_loc(images[rand_index[k]], n_seg=args.SuperP_size, compact=args.compact)
                    images[k, :, x_idxs, y_idxs] = images[rand_index[k], :, x_idxs, y_idxs]
                    lam.append(len(x_idxs) / (W * H))
            lam = torch.tensor(lam).cuda()
            _, preds = net(images)
            loss_batch = torch.mul(criterion(preds, target_b), lam) + torch.mul(criterion(preds, target_a), (1. - lam))
            loss = loss_batch.mean()
        else:
            _, preds = net(images)
            loss = criterion(preds, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_correct_tr += get_num_correct(preds,labels)

        del images;
        del labels;

    for batch in tqdm(test_loader):
        images, labels = batch
        images = images.float().cuda()
        labels = labels.long().cuda()

        net.eval()
        with torch.no_grad():
            feat4, out = net(images)
            pred_indices = torch.argmax(out, dim=1)
            correct = out.argmax(dim=1).eq(labels).sum().item()
            # print(pred_indices)
            # print(labels)

        loss = criterion(out, labels)

        test_loss += loss.item()
        total_correct_ts += correct

        del images;
        del labels;

    acc_tr = total_correct_tr / len(train_set)
    loss_tr = train_loss / len(train_set)
    loss_ts = test_loss / len(test_set)
    acc_ts = total_correct_ts / len(test_set)

    print('Ep: ', epoch, 'AC_tr: ', acc_tr, 'Loss_tr: ', loss_tr, 'AC_test: ', acc_ts, 'Loss_test: ', loss_ts)

    Acc_best2 = acc_ts

    with open(results_file_name, 'a') as file:
        file.write(
            'Epoch %d, train_acc = %.5f , train_loss = %.5f , test_acc = %.5f, test_loss = %.5f\n' % (
                epoch, acc_tr, loss_tr, acc_ts, loss_ts))

    if Acc_best2 >= Acc_best:
        Acc_best = Acc_best2
        epoch_best = epoch
        if len(args. gpu_ids) > 1:
            torch.save(net.module.state_dict(), net_name)
        else:
            torch.save(net.state_dict(), net_name)
        print('Best_Ep:', epoch, 'Best_test_Acc:', Acc_best)
print(' * Best_Ep:', epoch_best, ' * Best_test_Acc:', Acc_best)

