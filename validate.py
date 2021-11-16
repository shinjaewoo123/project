import torch
import torch.nn as nn
import timm
import os
import sys
from os.path import join as pjoin
from PIL import Image 
import pickle 
import albumentations
import numpy as np 
from glob import glob

from collections import defaultdict

from product_info_extract.data import get_dataloader
from product_info_extract.models import CustomModel
from product_info_extract.util import *

def load_model(args):
    model = CustomModel(args.modelname, args.columns, args.class_num_lst, feature_dim=args.feature_dim)
    model.load_state_dict(torch.load(pjoin(args.model_save_dir, 'best.pth')))
    model.to(args.device)
    model.eval()
    return model

def main():
    args = read_cfg()
    class_num_lst = args.class_num_lst
    device = args.device 

    ## load class_map
    with open(pjoin(args.data_dir, 'classes.pkl'), 'rb') as f:
        class_map = pickle.load(f)

    class_map_reverse = {}
    for column, dic in class_map.items():
        class_map_reverse[column] = {}
        for k, v in dic.items():
            class_map_reverse[column][v] = k
    # print(class_map_reverse) 

    model = load_model(args)

    train_dataset, val_dataset, train_dataloader, val_dataloader, class_num_lst = get_dataloader(
            args, columns=args.columns)
    
    ## get result
    result_dic = {column:defaultdict(list) for column in args.columns}
    result_dic['id_list'] = val_dataset.data['id'].tolist()

    with torch.no_grad():
        for i, (image, label) in enumerate(val_dataloader):
            image = image.to(device)
            label = [l.to(device) for l in label]
            batch_size = image.shape[0]

            pred = model(image)
            for idx, column in enumerate(args.columns):
                pred_column = nn.functional.softmax(pred[idx], dim=1)
                top_p, top_class = pred_column.topk(1, dim=1)
                
                pred_ret = [p[0] for p in top_class.cpu().numpy().tolist()]
                label_ret = label[idx].cpu().numpy().tolist()
                

                result_dic[column]['pred_list'].extend(pred_ret)
                result_dic[column]['gt_list'].extend(label_ret)
    
    for idx, column in enumerate(args.columns):
        result_dic[column]['class_map'] = sorted(class_map[column].keys())
    
    ## save result 
    with open(pjoin(args.model_save_dir, 'result.pkl'), 'wb') as f:
        pickle.dump(result_dic, f)
    
    run_metric(args)

    print('Done.')

if __name__ == "__main__":
    main()
