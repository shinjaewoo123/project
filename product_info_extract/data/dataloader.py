from PIL import Image 
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
import json
import torch
import torch.nn.functional as F
import os
import random
import tqdm

from os.path import join as pjoin
from glob import glob
import pandas as pd
import pickle

class AlbumentationDataset(Dataset):
    def __init__(self, data_dir, anno_filename, class_filename, columns, transform, istrain=True):
        """ """
        self.columns = columns
        self.image_dir = pjoin(data_dir, 'images')

        # read data
        self.data = pd.read_csv(pjoin(data_dir, anno_filename))
        self.data = self.data[self.data['istrain'] == istrain].reset_index(drop=True)
        with open(pjoin(data_dir, class_filename), 'rb') as f:
            self.classes = pickle.load(f)
        self.data = self.data[['id']+columns]

        self.transform = transform

    def __getitem__(self, idx):
        image = self.get_x(idx)
        label = self.get_y(idx)
        return image, label

    def __len__(self):
        return len(self.data)

    def get_x(self, idx):
        image = Image.open(pjoin(self.image_dir, f'{self.data["id"][idx]}.jpg')).convert('RGB')
        image = self.transform(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image, dtype=torch.float32)

    def get_y(self, idx):
        labels = []
        for column in self.columns:
            labels.append(torch.tensor(self.classes[column][self.data[column][idx]], dtype=torch.long))
        return labels


def get_dataloader(args, columns):
    """ """
    data_dir = args.data_dir
    anno_filename = args.anno_filename
    classes_filename = args.classes_filename
    resize_res = args.resize_res
    crop_res = args.crop_res
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_transform = albumentations.Compose([albumentations.Resize(resize_res, resize_res),
                                              albumentations.RandomCrop(crop_res, crop_res),
                                              albumentations.HorizontalFlip(),
                                              albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                                                                                val_shift_limit=10),
                                              albumentations.RandomBrightnessContrast(brightness_limit=0.1,
                                                                                      contrast_limit=0.1),
                                              albumentations.Normalize()
                                              ])

    val_transform = albumentations.Compose([albumentations.Resize(crop_res, crop_res),
                                            albumentations.Normalize()
                                            ])
    

    train_dataset = AlbumentationDataset(data_dir, anno_filename, classes_filename, columns, train_transform, istrain=True)
    val_dataset = AlbumentationDataset(data_dir, anno_filename, classes_filename, columns, val_transform, istrain=False)
    class_num_lst = [len(train_dataset.classes[c]) for c in train_dataset.columns]

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=False)

    return train_dataset, val_dataset, train_dataloader, val_dataloader, class_num_lst

