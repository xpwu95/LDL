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
# import torchvision
from timeit import default_timer as timer
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
from model.resnet50 import resnet50
import torch.backends.cudnn as cudnn
# torch.manual_seed(1)    # reproducible
from transforms.affine_transforms import *
import time
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil


# Hyper Parameters
EPOCH = 50               # train the training data n times, to save time, we just train 1 epoch
STEP_SIZE = 30
BATCH_SIZE = 1
BATCH_SIZE_TEST = 1
LR = 0.001              # learning rate
NUM_WORKERS = 8
NUM_CLASSES = 4
LOG_FILE_NAME = './logs/log_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '.log'
lr_steps = [30, 60, 90, 120]

np.random.seed(42)
# DATA_PATH = '/home/ubuntu3/cly/cly_web/sxx/data/sd/images/'
# TRAIN_FILE = '/home/ubuntu3/cly/cly_web/sxx/data/sd/train.txt'
# TEST_FILE = '/home/ubuntu3/cly/cly_web/sxx/data/sd/val.txt'

# log_file = open(LOG_FILE_NAME, 'a+')
# orig_stdout = sys.stdout
# sys.stdout = log_file

log = Logger()
log.open(LOG_FILE_NAME, mode="a")


data_dir = 'doctor-results/'
doctors = ['expert_a.txt', 'expertb2.txt', 'junior_a.txt', 'junior_b.txt', 'general_a.txt', 'general_b.txt', ]

cross_val_lists = ['0']#, '1', '2', '3', '4']

for i in range(3):

    imgs_a, preds_a = np.loadtxt(data_dir + doctors[2 * i], dtype=np.str, usecols=[1, 2]).T
    imgs_b, preds_b = np.loadtxt(data_dir + doctors[2 * i + 1], dtype=np.str, usecols=[1, 2]).T

    preds_a[np.where(preds_a == 'A')[0]] = 0
    preds_a[np.where(preds_a == 'B')[0]] = 1
    preds_a[np.where(preds_a == 'C')[0]] = 2
    preds_a[np.where(preds_a == 'D')[0]] = 3
    preds_a = preds_a.astype(int)

    preds_b[np.where(preds_b == 'A')[0]] = 0
    preds_b[np.where(preds_b == 'B')[0]] = 1
    preds_b[np.where(preds_b == 'C')[0]] = 2
    preds_b[np.where(preds_b == 'D')[0]] = 3
    preds_b = preds_b.astype(int)

    imgs = imgs_b  # np.hstack((imgs_a, imgs_b))
    preds = preds_b  # np.hstack((preds_a, preds_b))

    AVE_ACC = []
    Precision = []
    SE = []
    SP = []
    YI = []
    for cross_val_index in cross_val_lists:
        # log.write('\n\ncross_val_index: ' + cross_val_index + '\n\n')

        TRAIN_FILE = '/home/ubuntu3/wxp/datasets/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_trainval_' + cross_val_index + '.txt'
        TEST_FILE = '/home/ubuntu3/wxp/datasets/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_test_' + cross_val_index + '.txt'

        imgs_train, labels_train = np.loadtxt(TRAIN_FILE, dtype=np.str, usecols=[0, 1]).T
        labels_train = labels_train.astype(int)

        imgs_test, labels_test = np.loadtxt(TEST_FILE, dtype=np.str, usecols=[0, 1]).T
        labels_test = labels_test.astype(int)

        imgs_train_test = np.hstack((imgs_train, imgs_test))
        labels_train_test = np.hstack((labels_train, labels_test))

        # y_pred = preds[np.array([imgs.tolist().index(j) for j in imgs_test])]
        # y_true = labels_test
        # y_pred = preds[np.array([imgs.tolist().index(j) for j in imgs_train_test])]
        # y_true = labels_train_test

        y_true = labels_train_test[np.array([imgs_train_test.tolist().index(j) for j in imgs])]
        y_pred = preds

        Result, AVE_ACC_, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)

        Precision_, SE_, SP_, YI_ = Result[4]
        AVE_ACC.append(AVE_ACC_)
        Precision.append(Precision_)
        SE.append(SE_)
        SP.append(SP_)
        YI.append(YI_)

        log.write(str(pre_se_sp_yi_report) + '\n')

    log.write('pre:%.4f se:%.4f sp:%.4f yi:%.4f acc:%.4f\n' % (np.array(Precision).mean(), np.array(SE).mean(),
                                                         np.array(SP).mean(), np.array(YI).mean(),
                                                         np.array(AVE_ACC).mean()))
    log.write('pre:%.4f se:%.4f sp:%.4f yi:%.4f acc:%.4f\n' % (np.array(Precision).std(), np.array(SE).std(),
                                                         np.array(SP).std(), np.array(YI).std(),
                                                         np.array(AVE_ACC).std()))
    log.write('\n###########################################\n')

