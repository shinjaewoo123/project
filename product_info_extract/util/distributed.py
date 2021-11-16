import torch

def set_multinode(local_rank):
    """ """
    if local_rank != -1:
        device = torch.device('cuda:{}'.format(local_rank))
        torch.distributed.init_process_group(backend="nccl")
        global_rank = torch.distributed.get_rank()
    else:
        device = 'cuda'
        global_rank = -1
    
    return device, global_rank


