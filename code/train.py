# library
# standard library
import os, sys

# third-party library
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import dataset_processing
from timeit import default_timer as timer
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
from model.resnet50 import resnet50
import torch.backends.cudnn as cudnn
from transforms.affine_transforms import *
import time
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Hyper Parameters
BATCH_SIZE = 32
BATCH_SIZE_TEST = 20
LR = 0.001              # learning rate
NUM_WORKERS = 12
NUM_CLASSES = 4
LOG_FILE_NAME = './logs/log_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '.log'
lr_steps = [30, 60, 90, 120]

np.random.seed(42)

DATA_PATH = '/home/ubuntu5/wxp/datasets/acne4/VOCdevkit2007/VOC2007/JPEGImages_300'

log = Logger()
log.open(LOG_FILE_NAME, mode="a")


def criterion(lesions_num):
    if lesions_num <= 5:
        return 0
    elif lesions_num <= 20:
        return 1
    elif lesions_num <= 50:
        return 2
    else:
        return 3


def trainval_test(cross_val_index, sigma, lam):

    TRAIN_FILE = '/home/ubuntu5/wxp/datasets/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_trainval_' + cross_val_index + '.txt'
    TEST_FILE = '/home/ubuntu5/wxp/datasets/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_test_' + cross_val_index + '.txt'

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])

    dset_train = dataset_processing.DatasetProcessing(
        DATA_PATH, TRAIN_FILE, transform=transforms.Compose([
                transforms.Scale((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                RandomRotate(rotation_range=20),
                normalize,
            ]))

    dset_test = dataset_processing.DatasetProcessing(
        DATA_PATH, TEST_FILE, transform=transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))

    train_loader = DataLoader(dset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(dset_test,
                             batch_size=BATCH_SIZE_TEST,
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)

    cnn = resnet50().cuda()
    cudnn.benchmark = True

    params = []
    new_param_names = ['fc', 'counting']
    for key, value in dict(cnn.named_parameters()).items():
        if value.requires_grad:
            if any(i in key for i in new_param_names):
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]
            else:
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]

    optimizer = torch.optim.SGD(params, momentum=0.9)  #

    loss_func = nn.CrossEntropyLoss().cuda()
    kl_loss_1 = nn.KLDivLoss().cuda()
    kl_loss_2 = nn.KLDivLoss().cuda()
    kl_loss_3 = nn.KLDivLoss().cuda()

    def adjust_learning_rate_new(optimizer, decay=0.5):
        """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']

    # training and testing
    start = timer()
    test_acc_his = 0.7
    test_mae_his = 8
    test_mse_his = 18
    for epoch in range(lr_steps[-1]):#(EPOCH):#

        if epoch in lr_steps:
            adjust_learning_rate_new(optimizer, 0.5)
        # scheduler.step(epoch)

        losses_cls = AverageMeter()
        losses_cou = AverageMeter()
        losses_cou2cls = AverageMeter()
        losses = AverageMeter()
        # '''
        cnn.train()
        for step, (b_x, b_y, b_l) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

            b_x = b_x.cuda()
            b_l = b_l.numpy()

            # generating ld
            b_l = b_l - 1
            ld = genLD(b_l, sigma, 'klloss', 65)
            ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))).transpose()
            ld = torch.from_numpy(ld).cuda().float()
            ld_4 = torch.from_numpy(ld_4).cuda().float()

            # train
            cnn.train()

            cls, cou, cou2cls = cnn(b_x, None)nn output
            loss_cls = kl_loss_1(torch.log(cls), ld_4) * 4.0
            loss_cou = kl_loss_2(torch.log(cou), ld) * 65.0
            loss_cls_cou = kl_loss_3(torch.log(cou2cls), ld_4) * 4.0
            loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            losses_cls.update(loss_cls.item(), b_x.size(0))
            losses_cou.update(loss_cou.item(), b_x.size(0))
            losses_cou2cls.update(loss_cls_cou.item(), b_x.size(0))
            losses.update(loss.item(), b_x.size(0))
        message = '%s %6.0f | %0.3f | %0.3f | %0.3f | %0.3f | %s\n' % ( \
                "train", epoch,
                losses_cls.avg,
                losses_cou.avg,
                losses_cou2cls.avg,
                losses.avg,
                time_to_str((timer() - start), 'min'))
        # print(message)
        log.write(message)
        # '''
        if epoch >= 9:
            with torch.no_grad():
                test_loss = 0
                test_corrects = 0
                y_true = np.array([])
                y_pred = np.array([])
                y_pred_m = np.array([])
                l_true = np.array([])
                l_pred = np.array([])
                cnn.eval()
                for step, (test_x, test_y, test_l) in enumerate(test_loader):   # gives batch data, normalize x when iterate train_loader

                    test_x = test_x.cuda()
                    test_y = test_y.cuda()

                    y_true = np.hstack((y_true, test_y.data.cpu().numpy()))
                    l_true = np.hstack((l_true, test_l.data.cpu().numpy()))

                    cnn.eval()

                    cls, cou, cou2cls = cnn(test_x, None)

                    loss = loss_func(cou2cls, test_y)
                    test_loss += loss.data

                    _, preds_m = torch.max(cls + cou2cls, 1)
                    _, preds = torch.max(cls, 1)
                    # preds = preds.data.cpu().numpy()
                    y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))
                    y_pred_m = np.hstack((y_pred_m, preds_m.data.cpu().numpy()))

                    _, preds_l = torch.max(cou, 1)
                    preds_l = (preds_l + 1).data.cpu().numpy()
                    # preds_l = cou2cou.data.cpu().numpy()
                    l_pred = np.hstack((l_pred, preds_l))

                    batch_corrects = torch.sum((preds == test_y)).data.cpu().numpy()
                    test_corrects += batch_corrects

                test_loss = test_loss.float() / len(test_loader)
                test_acc = test_corrects / len(test_loader.dataset)#3292  #len(test_loader)
                message = '%s %6.1f | %0.3f | %0.3f\n' % ( \
                        "test ", epoch,
                        test_loss.data,
                        test_acc)

                _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)
                _, _, pre_se_sp_yi_report_m = report_precision_se_sp_yi(y_pred_m, y_true)
                _, MAE, MSE, mae_mse_report = report_mae_mse(l_true, l_pred, y_true)

                if True:
                    log.write(str(pre_se_sp_yi_report) + '\n')
                    log.write(str(pre_se_sp_yi_report_m) + '\n')
                    log.write(str(mae_mse_report) + '\n')


cross_val_lists = ['0', '1', '2', '3', '4']
for cross_val_index in cross_val_lists:
    log.write('\n\ncross_val_index: ' + cross_val_index + '\n\n')
    if True:
        trainval_test(cross_val_index, sigma=30 * 0.1, lam=6 * 0.1)
