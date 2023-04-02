from toad.utils.decorator import support_numpy

def flooding(loss, b):
    """flooding loss
    """
    return (loss - b).abs() + b


@support_numpy
def focal_loss(input, target, alpha = 1., gamma = 2., reduction = 'mean'):
    """focal loss
    
    Args:
        input (Tensor): N x C, C is the number of classes
        target (Tensor): N, each value is the index of classes
        alpha (Variable): balaced variant of focal loss, range is in [0, 1]
        gamma (float): focal loss parameter
        reduction (str): `mean`, `sum`, `none` for reduce the loss of each classes
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    prob = F.sigmoid(input)
    weight = torch.pow(1. - prob, gamma)
    focal = -alpha * weight * torch.log(prob)
    loss = F.nll_loss(focal, target, reduction = reduction)

    return loss


@support_numpy
def binary_focal_loss(input, target, **kwargs):
    """binary focal loss
    """
    # convert 1d tensor to 2d
    if input.ndim == 1:
        import torch
        input = input.view(-1, 1)
        input = torch.hstack([1 - input, input])
    
    return focal_loss(input, target, **kwargs)


def focal_loss_for_numpy(input, target, alpha = 1., gamma = 2., reduction = 'mean'):
    """focal loss for numpy array
    """
    import numpy as np

    prob = 1 / (1 + np.exp(-input))
    weight = np.power(1. - prob, gamma)
    focal = -alpha * weight * np.log(prob)
    loss = -focal[np.arange(len(focal)), target]

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass

    return loss


def label_smoothing(labels, smoothing = 0.1):
    """label smoothing
    """
    assert len(labels.shape) == 2, "labels must be 2 dim where shape should be (N, C)"

    return (1. - smoothing) * labels + smoothing / labels.shape[1]
