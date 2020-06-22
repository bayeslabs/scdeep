import numpy as np
import pandas as pd
import torch
from torch import nn
import scipy.sparse

from utils import nan2inf, reduce_mean


def nb_loss(y_true, output, mean=True, eps=1e-10, scale_factor=1.0, ridge_lambda=None, mask=False):
    y_pred, theta = output

    y_true = y_true.type(torch.FloatTensor)
    y_pred = y_pred.type(torch.FloatTensor) * scale_factor

    theta = torch.min(theta, torch.zeros_like(theta) + 1e6)
    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))

    loss = t1 + t2

    loss = nan2inf(loss)

    if mean:
        if mask:
            loss = reduce_mean(loss)
        else:
            loss = torch.mean(loss)

    return loss


def zinb_loss(y_true, output, mean=True, eps=1e-10, scale_factor=1.0, ridge_lambda=0., mask=False):
    y_pred, theta, pi = output

    nb_case = nb_loss(y_true, [y_pred, theta], mean=False, eps=eps, scale_factor=scale_factor) - torch.log(1.0 - pi + eps)

    y_true = y_true.type(torch.FloatTensor)
    y_pred = y_pred.type(torch.FloatTensor) * scale_factor
    theta = torch.min(theta, torch.zeros_like(theta) + 1e6)

    zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -(torch.log(pi + ((1.0 - pi) * zero_nb) + eps))
    result = torch.where((y_true < 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * torch.square(pi)
    result += ridge

    result = nan2inf(result)

    if mean:
        if mask:
            result = reduce_mean(result)
        else:
            result = torch.mean(result)
    return result
