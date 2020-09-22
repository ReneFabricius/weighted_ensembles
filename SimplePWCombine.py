import torch
from timeit import default_timer as timer
import numpy as np
import os


def m1(PP):
    n, k, kk = PP.size()
    assert k == kk

    E = torch.eye(k).cuda()
    Es = E.unsqueeze(0).expand(n, k, k)
    B = torch.zeros(n, k, 1).cuda()

    A = (PP.sum(dim=2).diag_embed() + PP) / (k - 1) - Es
    A[:, k - 1, :] = 1

    Xs, LUs = torch.solve(B, A)
    ps = Xs[:, 0:k, 0:1].squeeze(2)
    return ps


def m2(PP):
    n, k, kk = PP.size()
    assert k == kk

    es = torch.ones(n, k, 1, dtype=torch.get_default_dtype()).cuda()
    zs = torch.zeros(n, 1, 1).cuda()
    B = torch.zeros(n, k + 1, 1).cuda()

    Q = (PP * PP).sum(dim=1).diag_embed() - PP * PP.transpose(1, 2)
    A = torch.cat((Q, es), 2)
    A = torch.cat((A, torch.cat((es.transpose(1, 2), zs), 2)), 1)
    Xs, LUs = torch.solve(B, A)
    ps = Xs[:, 0:k, 0:1].squeeze(2)
    return ps