import tqdm
import torch

def train_one_epoch(model, train_dataloader, optim, criterion, epoch, logger, scheduler, device, writer):
    
    model.train()
    columns = model.columns
    class_num_lst = model.class_num_lst
    loss_list, acc_sum = [], 0
    max_iter = len(train_dataloader)
    progress_bar = tqdm.tqdm(train_dataloader)
    acc1_list_collection = [[] for _ in range(len(columns))]
    acc2_list_collection = [[] for _ in range(len(columns))]
    total_iter = 0
    for i, (image, label) in enumerate(progress_bar):
        image = image.to(device)
        label = [l.to(device) for l in label]
        batch_size = image.shape[0]

        optim.zero_grad()
        pred = model(image)

        loss = criterion(pred, label)
        loss.backward()
        optim.step()
        loss_list.append(loss.item())

        total_iter = len(loss_list)
        
        # calculate acc each class
        for idx, class_num in enumerate(class_num_lst):
            k = class_num if class_num < 2 else 2
            acc1, acc2 = accuracy(pred[idx], label[idx], topk=(1, k))
            acc1_list_collection[idx].append(acc1.item())
            acc2_list_collection[idx].append(acc2.item())

        progress_bar_text = f'Epoch [{epoch}] Loss [{round(loss.item(), 5)}] '
        for idx, column in enumerate(columns):
            progress_bar_text += f'{column} "Acc1" [{round(acc1_list_collection[idx][i] / batch_size, 2)}] '
            progress_bar_text += f'{column} "Acc2" [{round(acc2_list_collection[idx][i] / batch_size, 2)}] ' 
        progress_bar.set_description(progress_bar_text)
        progress_bar.update()

    loss_mean = sum(loss_list) / max_iter
    writer.add_scalar('train/loss', loss_mean, str(epoch))
    logger.info(f'epoch = {str(epoch)}, train/loss = {loss_mean}')
    for idx, column in enumerate(columns):
        acc1_mean = round(sum(acc1_list_collection[idx]) / (max_iter * batch_size), 2)
        acc2_mean = round(sum(acc2_list_collection[idx]) / (max_iter * batch_size), 2)
        writer.add_scalar(f'train/{column}_acc1', acc1_mean, str(epoch))
        writer.add_scalar(f'train/{column}_acc2', acc2_mean, str(epoch))
        logger.info(f'epoch = {str(epoch)}, train/{column}_acc1 = {acc1_mean}')
        logger.info(f'epoch = {str(epoch)}, train/{column}_acc2 = {acc2_mean}')
    
    progress_bar.update()


def validate_one_epoch(model, val_dataloader, criterion, epoch, logger, device, writer):

    model.eval()
    columns = model.columns
    class_num_lst = model.class_num_lst
    acc1_list_collection = [[] for _ in range(len(columns))]
    acc2_list_collection = [[] for _ in range(len(columns))]
 
    loss_list, acc_sum = [], 0
    max_iter = len(val_dataloader)
    total_iter = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(val_dataloader):
            image = image.to(device)
            label = [l.to(device) for l in label]
            batch_size = image.shape[0]

            pred = model(image)
            loss = criterion(pred, label)
            loss_list.append(loss.item())
            
            # calculate acc each class
            for idx, class_num in enumerate(class_num_lst):
                k = class_num if class_num < 2 else 2
                acc1, acc2 = accuracy(pred[idx], label[idx], topk=(1, k))
                acc1_list_collection[idx].append(acc1.item())
                acc2_list_collection[idx].append(acc2.item())
            total_iter += batch_size 

    loss_mean = round(sum(loss_list) / max_iter, 5) 
    print_text = f'Epoch [{epoch}] Val_Loss [{loss_mean}] '
    writer.add_scalar('val/loss', loss_mean, str(epoch))
    logger.info(f'epoch = {str(epoch)}, val/loss = {loss_mean}')
    for idx, column in enumerate(columns):
        acc1_mean = round(sum(acc1_list_collection[idx]) / total_iter, 2)
        acc2_mean = round(sum(acc2_list_collection[idx]) / total_iter, 2)    

        print_text += f'{column} "Acc1" [{acc1_mean}] '
        print_text += f'{column} "Acc2" [{acc2_mean}] '
        writer.add_scalar(f'val/{column}_acc1', acc1_mean, str(epoch))
        writer.add_scalar(f'val/{column}_acc2', acc2_mean, str(epoch))
        logger.info(f'epoch = {str(epoch)}, val/{column}_acc1 = {acc1_mean}')
        logger.info(f'epoch = {str(epoch)}, val/{column}_acc2 = {acc2_mean}')

    logger.info('='*20)
    print(print_text)
    
    return sum(loss_list) / (total_iter)


def accuracy(output, target, topk=(1, 2)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0))

        return res

