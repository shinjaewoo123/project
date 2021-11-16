import pickle 
import numpy as np 
import pandas as pd

import os 
from os.path import join as pjoin
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, precision_score


def convert_to_binary(gt_list, pred_list, cls):
    return np.array(gt_list) == cls, np.array(pred_list) == cls


def specifi_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    total_n = (y_true == 0).sum()
    tn = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1

    return tn / total_n


def calculate_metric(gt_list, pred_list, cls, cls_list):
    """docstring for calculate_metric"""
    gt, pred = convert_to_binary(gt_list, pred_list, cls)
    
    mat = confusion_matrix(gt, pred)
    acc = accuracy_score(gt, pred)
    pre = precision_score(gt, pred)
    rec = recall_score(gt, pred)
    spe = specifi_score(gt, pred)
    f1 = f1_score(gt, pred)

    print('cls :', cls_list[cls])
    print('정확도 : ', acc)
    print('양성예측도 : ', pre)
    print('민감도 : ', rec)
    print('특이도 : ', spe)
    print('f1 score : ', f1)
    print('')
    

def run_metric(args):
    with open(pjoin(args.model_save_dir, 'result.pkl'), 'rb') as f:
        all_data = pickle.load(f)

    for column in args.columns:
        data = all_data[column]
        gt_list, pred_list, cls_list = data['gt_list'], data['pred_list'], data['class_map']

        print(f'== {column} metric == ')
        print(confusion_matrix(gt_list, pred_list))

        for cls_idx in range(len(cls_list)):
            calculate_metric(gt_list, pred_list, cls_idx, cls_list)
