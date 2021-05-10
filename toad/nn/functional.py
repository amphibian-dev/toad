import torch.nn.functional as F

def flooding(loss, b):
    """flooding loss
    """
    return (loss - b).abs() + b


def focalloss(input, target, alpha = 1., gamma = 2., reduction = 'mean'):
    """focal loss
    Args:
        input (Tensor): N x C, C is the number of classes
        target (Tensor): N, each value is the index of classes
        alpha (Variable): balaced variant of focal loss, range is in [0, 1]
        gamma (float): focal loss parameter
        reduction (str): `mean`, `sum`, `none` for reduce the loss of each classes
    """
    prob = F.sigmoid(input, dim = 1)
    weight = torch.pow(1. - prob, gamma)
    focal = -alpha * weight * torch.log(prob)
    loss = F.nll_loss(focal, target, reduction = reduction)
    return loss
