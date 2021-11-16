import os
import torch


def create_model_save_dir(save_dir):
    """ """
    model_save_dir=''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    folder_list = os.listdir(save_dir)
    if len(folder_list) > 0:
        folder_list = list(folder_list)
        folder_list = list(map(int, folder_list))
        folder_list.sort()
        last_folder = folder_list[-1]
        last_number = int(last_folder)
        new_folder = str(last_number + 1)
    else:
        new_folder = '1'
    model_save_dir = os.path.join(save_dir, new_folder)
    os.makedirs(model_save_dir)

    return model_save_dir
