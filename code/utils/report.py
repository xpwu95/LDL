# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:04:18 2018

@author: Kunimitsu Tezuka
"""

import numpy as np
from sklearn.metrics import accuracy_score
from .metrics import sensitivity_specificity_support, specificity_score, sensitivity_score


def creat_class_name(classNumber):
    number = []
    for i in range(classNumber):
        number.append('class'+str(i))
    return number


def report_precision_se_sp_yi(y_predicitions,  groundture):
    class_list1 = np.unique(groundture)
    Result = []
    SE, SP, _ = sensitivity_specificity_support(groundture, y_predicitions)
    YI = SE + SP - 1
    for i in range(class_list1.shape[0]):
        local_1 = [k for k in range(len(y_predicitions)) if y_predicitions[k]==class_list1[i]]
        y_pred1 = [y_predicitions[k] for k in local_1]
        y_true1 = [groundture[k] for k in local_1]
        pre = accuracy_score(y_true1,y_pred1)
        Result.append([pre,SE[i],SP[i],YI[i]])
    AVE_ACC = accuracy_score(groundture,y_predicitions)
    AVE_Pre = np.mean(Result,0)[0]
    AVE_SE = sensitivity_score(groundture,y_predicitions,average='macro')
    AVE_SP = specificity_score(groundture,y_predicitions,average='macro')
    AVE_YI = AVE_SE+AVE_SP-1
    Result.append([AVE_Pre,AVE_SE,AVE_SP,AVE_YI])
    target_names = creat_class_name(len(Result)-1)
    last_line_heading = 'avg / total'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), 3)
    target_names.append(last_line_heading)
    headers = ["Precision", "SE", "SP", "YI"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += " "
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    for i in range(len(Result)):
        fmt1 = '%% %ds' % width
        fmt1 += " "
        fmt1 += ' '.join(['% 9s' for _ in Result[i]])
        fmt1 += '\n'
        values = []
        for j in (Result[i]):
            values+=["{0:0.{1}f}".format(j, 4)]
        tmp = fmt1%tuple([target_names[i]]+values)
        report+=tmp
    head = ['AVE_ACC']
    fmt1 = '%% %ds' % width
    fmt1 += " "
    fmt1 += ' '.join(['% 9s' for _ in head])
    fmt1 += '%% %ds'%width
    fmt1 += ' '.join(['%s' for _ in [AVE_ACC]])
    value = ["{0:0.{1}f}".format(AVE_ACC, 4)]
    report += fmt1%tuple(["",'AVE_ACC','']+value)
    report += '\n'
    # print(report)
    return Result, AVE_ACC, report

def report_mae_mse(y_ture, y_predicitions,classification):
    classed = np.unique(classification)
    target_name = creat_class_name(classed.shape[0])
    target_name.append('avg / total')
    name_width = max(len(cn) for cn in target_name)
    headers = ['MAE','MSE']
    fmt = '%% %ds' % name_width  # first column: class name
    fmt += " "
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = ['']+headers
    report = fmt%tuple(headers)
    Result = []
    for i in range(classed.shape[0]):
        local_list = [k for k in range(len(classification)) if classification[k]==classed[i]]
        y_ture_class_i = [y_ture[j] for j in local_list]
        y_predicitions_class_i = [y_predicitions[k] for k in local_list]
        w =np.array(y_predicitions_class_i)-np.array(y_ture_class_i)
        MAE = np.mean(np.abs(w))
        MSE = np.sqrt(np.mean(w*w))
        Result.append([MAE,MSE])
        fmt1 = '%% %ds' % name_width
        fmt1 += " "
        fmt1 += ' '.join(['% 9s' for _ in Result[i]])
        fmt1 += '\n'
        values = []
        for j in (Result[i]):
            values += ["{0:0.{1}f}".format(j, 4)]
        tmp = fmt1 % tuple([target_name[i]] + values)
        report += tmp
    w = np.array(y_ture) - np.array(y_predicitions)
    MAE = np.mean(np.abs(w))
    MSE = np.sqrt(np.mean(w * w))
    Result.append([MAE, MSE])
    fmt1 = '%% %ds' % name_width
    fmt1 += " "
    fmt1 += ' '.join(['% 9s' for _ in Result[i]])
    fmt1 += '\n'
    values = []
    i = len(Result)-1
    for j in (Result[i]):
        values += ["{0:0.{1}f}".format(j, 4)]
    tmp = fmt1 % tuple([target_name[i]] + values)
    report += tmp
    # print(report)
    return Result, MAE, MSE, report

