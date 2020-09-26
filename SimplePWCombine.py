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
    B[:, k - 1, :] = 1

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
    B[:, k, :] = 1

    Q = (PP * PP).sum(dim=1).diag_embed() - PP * PP.transpose(1, 2)
    A = torch.cat((Q, es), 2)
    A = torch.cat((A, torch.cat((es.transpose(1, 2), zs), 2)), 1)
    Xs, LUs = torch.solve(B, A)
    ps = Xs[:, 0:k, 0:1].squeeze(2)
    return ps


def bc(PP):
    n, k, kk = PP.size()
    assert k == kk
    eps = 1e-5

    MMi = (1/k)*(torch.eye(k - 1) + torch.ones(k - 1, k - 1)).cuda()
    rws = int(k * (k - 1) / 2)
    M = torch.zeros(rws, k - 1)
    for c in range(k - 1):
        rs = int((c + 1) * k - (c + 1) * (c + 2) / 2)
        re = int((c + 2) * k - (c + 2) * (c + 3) / 2)
        M[rs:re, c] = -1
        oi = c
        cs = 0
        while oi >= 0:
            M[cs + oi, c] = 1
            oi -= 1
            cs += k - (c - oi)

    M = M.cuda()
    MMiM = torch.matmul(MMi, M.T)

    rv = PP[:, torch.triu(torch.ones(k, k).cuda(), 1) == 1].T
    small = eps*torch.ones(rv.size()).cuda()
    rv = torch.where(rv < eps, small, rv)
    rv = torch.where(rv > 1 - eps, 1 - small, rv)
    s = torch.log(1 / rv - 1)
    u = torch.matmul(MMiM, s)
    zs = torch.zeros(1, n)
    u_exp = torch.exp(torch.cat([zs.cuda(), u], dim=0))
    ps = u_exp / torch.sum(u_exp, dim=0)

    return ps.T