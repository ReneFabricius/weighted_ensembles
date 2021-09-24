import torch
from timeit import default_timer as timer
import numpy as np
import os


def m1(PP, device=torch.device("cuda")):
    n, k, kk = PP.size()
    assert k == kk

    E = torch.eye(k, device=device)
    Es = E.unsqueeze(0).expand(n, k, k)
    B = torch.zeros(n, k, 1, device=device)
    B[:, k - 1, :] = 1

    A = (PP.sum(dim=2).diag_embed() + PP) / (k - 1) - Es
    A[:, k - 1, :] = 1

    Xs, LUs = torch.solve(B, A)
    ps = Xs[:, 0:k, 0:1].squeeze(2)
    return ps


def m2(PP, device=torch.device("cuda")):
    n, k, kk = PP.size()
    assert k == kk

    es = torch.ones(n, k, 1, dtype=torch.get_default_dtype(), device=device)
    zs = torch.zeros(n, 1, 1, device=device)
    B = torch.zeros(n, k + 1, 1, device=device)
    B[:, k, :] = 1

    Q = (PP * PP).sum(dim=1).diag_embed() - PP * PP.transpose(1, 2)
    A = torch.cat((Q, es), 2)
    A = torch.cat((A, torch.cat((es.transpose(1, 2), zs), 2)), 1)
    Xs, LUs = torch.solve(B, A)
    ps = Xs[:, 0:k, 0:1].squeeze(2)
    return ps


def m2_iter(PP, device=torch.device("cuda")):
    n, k, kk = PP.size()
    assert k == kk
    max_iter = max(100, k)  # As per LIBSVM implementation
    eps = 0.005 / k         # As per LIBSVM implementation

    PP = PP.to(device=device, dtype=torch.float64)
    Q = (PP * PP).sum(dim=1).diag_embed() - PP * PP.transpose(1, 2)
    p = torch.ones(n, k, 1, device=device, dtype=torch.float64) / k

    for it in range(max_iter):
        Qp = torch.matmul(Q, p)
        pQp = torch.matmul(p.transpose(1, 2), Qp)

        max_err = torch.max(torch.abs(Qp - pQp)).item()
        if max_err < eps:
            break

        for t in range(k):
            diff = (-Qp[:, [t]] + pQp) / Q[:, [t]][:, :, [t]]
            p[:, [t]] += diff
            pQp = (pQp + diff * (diff * Q[:, [t]][:, :, [t]] + 2 * Qp[:, [t]])) / (1 + diff) / (1 + diff)
            Qp = (Qp + diff * Q[:, :, [t]]) / (1 + diff)

    return p.squeeze(2)


def bc(PP, device=torch.device("cuda")):
    n, k, kk = PP.size()
    assert k == kk
    eps = 1e-5

    MMi = (1/k)*(torch.eye(k - 1, device=device) + torch.ones(k - 1, k - 1, device=device))
    rws = int(k * (k - 1) / 2)
    # Mapping h is used such that elements of {1, ..., k(k-1)/2}
    # are placed into upper triangle of k x k matrix row by row from left to right.
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

    rv = PP[:, torch.triu(torch.ones(k, k, device=device), 1) == 1].T
    small = eps*torch.ones(rv.size(), device=device)
    rv = torch.where(rv < eps, small, rv)
    rv = torch.where(rv > 1 - eps, 1 - small, rv)
    s = torch.log(1 / rv - 1)
    u = torch.matmul(MMiM, s)
    zs = torch.zeros(1, n)
    u_exp = torch.exp(torch.cat([zs.cuda(), u], dim=0))
    ps = u_exp / torch.sum(u_exp, dim=0)

    return ps.T