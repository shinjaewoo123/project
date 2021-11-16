import torch
import torch.nn as nn
import timm
from tensorboardX import SummaryWriter
import os
import sys
from os.path import join as pjoin

from product_info_extract.data import get_dataloader
from product_info_extract.models import CustomModel
from product_info_extract.loss import create_mh_loss_func
from product_info_extract.util import *

def main():
    # 0. yaml 파일을 읽어와 학습 파라미터들 설정
    args = read_cfg()
    
    # 1. 결과파일 저장경로 설정 
    model_save_dir = create_model_save_dir(args.save_dir)
    args.model_save_dir = model_save_dir
    writer = SummaryWriter(pjoin(model_save_dir, 'tb'))

    logger = LoggingWrapper(__name__)
    logger.add_file_handler(pjoin(model_save_dir, 'log'), LoggingWrapper.DEBUG, None)
    logger.add_stream_handler(sys.stdout, LoggingWrapper.DEBUG, None)
    logger.info('Main module is initialized.')

    # 2. 랜덤 시드 고정 & device 설정 
    set_random_seeds(args.random_seed)
    device = args.device 
    
    # 3. 데이터 로더 준비
    train_dataset, val_dataset, train_dataloader, val_dataloader, class_num_lst = get_dataloader(
            args, columns=args.columns)
    args.class_num_lst = class_num_lst
    logger.info(f'number of train data {len(train_dataloader)}')
    logger.info(f'number of val data {len(val_dataloader)}')

    # 4. cfg 정보 업데이트 및 저장.
    write_cfg(args)

    # 5. 모델 로드
    model = CustomModel(args.modelname, args.columns, class_num_lst, args.feature_dim)
    model.to(device)
    print(model)
    

    # 6. 학습 로스함수, 옵티마이저 설정
    criterion = create_mh_loss_func(len(args.columns))
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

    # 7. 모델 학습
    # best_loss = 1.
    best_loss = 100.
    for epoch in range(args.num_epoch):
        train_one_epoch(model, train_dataloader, optim, criterion, epoch, logger, scheduler=scheduler, device=device, writer=writer)
        val_loss = validate_one_epoch(model, val_dataloader, criterion, epoch, logger, device=device, writer=writer)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            fn = os.path.join(model_save_dir, 'best.pth')
            torch.save(model.state_dict(), fn)
            best_loss = val_loss
        # fn = os.path.join(model_save_dir, 'last.pth')
        fn = os.path.join(model_save_dir, f'{epoch}.pth')
        torch.save(model.state_dict(), fn)
        model.train()


if __name__ == "__main__":
    main()
