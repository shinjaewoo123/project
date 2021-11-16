import torch
import torch.nn as nn
import timm

class CustomModel(nn.Module):
    def __init__(self, basemodel, columns, class_num_lst, feature_dim=2048):
        super(CustomModel, self).__init__()
        self.model = timm.create_model(basemodel, pretrained=True)
        self.columns = columns
        self.class_num_lst = class_num_lst

        # create multihead layers
        for i, class_num in enumerate(self.class_num_lst):
            layer_list = []
            layer_list.append(nn.Conv2d(feature_dim, 512, 3)) 
            layer_list.append(nn.BatchNorm2d(512))
            layer_list.append(nn.SiLU())
            layer_list.append(nn.AdaptiveAvgPool2d((1,1)))
            layer_list.append(nn.Flatten(1))
            layer_list.append(nn.Linear(512, 512))
            layer_list.append(nn.Linear(512, class_num))
            setattr(self, f'seq_{i}', nn.Sequential(*layer_list))

    def forward(self, x):
        x = self.model.forward_features(x)
        return [getattr(self, f'seq_{i}')(x) for i, class_num in enumerate(self.class_num_lst)]

