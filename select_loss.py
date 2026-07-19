import torch
import torch.nn.functional as F
import math
import numpy as np

class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=9, gpu=True):
        super(NCC, self).__init__()
        self.win = win
        if gpu:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    def forward(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], (
            "volumes should be 1 to 3 dimensions. found: %d" % ndims
        )

        # set window size
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = 1
            padding = pad_no
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


def CharbonnierLoss(predict, target, eps=1e-3):
    return torch.mean(torch.sqrt((predict - target) ** 2 + eps**2))

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss (works for 2D or 3D flow).
    """

    def __init__(self, penalty="l1", loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        # y_pred: (B, C, H, W) or (B, C, D, H, W)
        ndims = y_pred.ndim - 2
        grads = []
        if ndims >= 1:
            # last spatial axis
            dx = torch.abs(y_pred[..., 1:] - y_pred[..., :-1])
            grads.append(dx * dx if self.penalty == "l2" else dx)
        if ndims >= 2:
            dy = torch.abs(y_pred[..., 1:, :] - y_pred[..., :-1, :])
            grads.append(dy * dy if self.penalty == "l2" else dy)
        if ndims >= 3:
            dz = torch.abs(y_pred[..., 1:, :, :] - y_pred[..., :-1, :, :])
            grads.append(dz * dz if self.penalty == "l2" else dz)

        grad = sum(torch.mean(g) for g in grads) / float(len(grads))
        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class L1_norm(torch.nn.Module):
    def __init__(self):
        super(L1_norm, self).__init__()

    def forward(self, img1):
        return torch.mean(torch.abs(img1))
