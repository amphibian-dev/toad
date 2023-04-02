import pytest

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .functional import focal_loss, binary_focal_loss

DATASET_SIZE = 20000
NUM_CLASSES = 4


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    yield


def test_focal_loss(seed):
    y_pred = torch.rand(DATASET_SIZE, NUM_CLASSES, dtype=torch.float)
    y = torch.randint(NUM_CLASSES, size=(DATASET_SIZE,), dtype=torch.long)
    loss = focal_loss(y_pred, y)
    assert loss.item() == pytest.approx(-0.07764504849910736, 1e-6)


def test_loss_with_grad(seed):
    y_pred = torch.rand(DATASET_SIZE, NUM_CLASSES, dtype=torch.float, requires_grad=True)
    y = torch.randint(NUM_CLASSES, size=(DATASET_SIZE,), dtype=torch.long)
    loss = focal_loss(y_pred, y)
    loss.backward()
    assert y_pred.grad is not None


def test_binary_focal_loss(seed):
    y_pred = torch.rand(DATASET_SIZE, dtype=torch.float)
    y = torch.randint(2, size=(DATASET_SIZE,), dtype=torch.long)
    loss = binary_focal_loss(y_pred, y)
    assert loss.item() == pytest.approx(-0.07776755839586258, 1e-6)


def test_numpy_support_focal_loss(seed):
    y_pred = torch.rand(DATASET_SIZE, NUM_CLASSES, dtype=torch.float).numpy()
    y = torch.randint(NUM_CLASSES, size=(DATASET_SIZE,), dtype=torch.long).numpy()
    loss = focal_loss(y_pred, y)
    assert loss.item() == pytest.approx(-0.07764504849910736, 1e-6)


def test_binary_focal_loss_for_xgb(seed):
    from toad.utils.decorator import xgb_loss

    y_pred = torch.rand(DATASET_SIZE, dtype=torch.float).numpy()
    y = torch.randint(2, size=(DATASET_SIZE,), dtype=torch.long).numpy()
    loss_func = xgb_loss(gamma=5.0, alpha=0.5)(binary_focal_loss)
    grad, hess = loss_func(y_pred, y)

    assert grad == pytest.approx(-0.00023283064365386963)
    assert hess == pytest.approx(465.66128730773926)


# TODO
# focal loss sum/none
# label_smoothing
