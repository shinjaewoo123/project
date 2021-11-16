import torch
import timm

def create_mh_loss_func(n, using_mixup=False):
    """ """
    if using_mixup:
        loss_fn = timm.loss.SoftTargetCrossEntropy
    else:
        loss_fn = torch.nn.CrossEntropyLoss

    def mh_loss(outputs, targets):
        res = None
        for i in range(n):
            if res:
                res += loss_fn()(outputs[i], targets[i])
            else:
                res = loss_fn()(outputs[i], targets[i])
        return res / n

    return mh_loss
