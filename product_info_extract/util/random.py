import random
import numpy as np
import torch

def set_random_seeds(random_seed=1):

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi gpu
    torch.backends.cudnn.deterministic = True # 
    torch.backends.cudnn.benchmark = True # if want fix rand, set False
    np.random.seed(random_seed)
    random.seed(random_seed)

