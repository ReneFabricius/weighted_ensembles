import torch
from timeit import default_timer as timer
import numpy as np
import os


def m1(PP, verbose=False):
    device = PP.device
    dtype = PP.dtype
    n, k, kk = PP.size()
    assert k == kk
    if verbose:
        print("Working with {} samples, each with {} classes".format(n, k))
        print("Solving for pairwise probabilities\n{}".format(PP.cpu().numpy()))

    E = torch.eye(k, device=device, dtype=dtype)
    Es = E.unsqueeze(0).expand(n, k, k)
    B = torch.zeros(n, k, 1, device=device, dtype=dtype)
    B[:, k - 1, :] = 1

    A = (PP.sum(dim=2).diag_embed() + PP) / (k - 1) - Es
    A[:, k - 1, :] = 1

    if verbose:
        print("Solving linear system\n{}\n× x =\n{}".format(A.cpu().numpy(), B.cpu().numpy()))

    Xs, LUs = torch.solve(B, A)
    ps = Xs[:, 0:k, 0:1].squeeze(2)

    if verbose:
        print("Resulting probabilities\n{}".format(ps.cpu().numpy()))

    return ps


def m2(PP, verbose=False):
    device = PP.device
    dtype = PP.dtype
    n, k, kk = PP.size()
    assert k == kk

    es = torch.ones(n, k, 1, dtype=dtype, device=device)
    zs = torch.zeros(n, 1, 1, device=device, dtype=dtype)
    B = torch.zeros(n, k + 1, 1, device=device, dtype=dtype)
    B[:, k, :] = 1

    Q = (PP * PP).sum(dim=1).diag_embed() - PP * PP.transpose(1, 2)
    A = torch.cat((Q, es), 2)
    A = torch.cat((A, torch.cat((es.transpose(1, 2), zs), 2)), 1)
    Xs, LUs = torch.solve(B, A)
    ps = Xs[:, 0:k, 0:1].squeeze(2)

    return ps


def m2_iter(PP, verbose=False):
    device = PP.device
    dtype = PP.dtype
    n, k, kk = PP.size()
    assert k == kk
    max_iter = max(100, k)  # As per LIBSVM implementation
    eps = 1e-12 / k
    min_prob = (1e-16 if dtype == torch.float64 else 1e-7)  # As per LIBSVM
    not_diag = torch.logical_xor(torch.ones(k, k, dtype=torch.bool, device=device),
                                 torch.eye(k, dtype=torch.bool, device=device))

    PP[(PP < min_prob) & not_diag] = min_prob
    PP[(PP > (1 - min_prob)) & not_diag] = 1 - min_prob

    Q = (PP * PP).sum(dim=1).diag_embed() - PP * PP.transpose(1, 2)
    p = torch.ones(n, k, 1, device=device, dtype=dtype) / k

    for it in range(max_iter):
        Qp = torch.matmul(Q, p)
        pQp = torch.matmul(p.transpose(1, 2), Qp)

        max_err = torch.max(torch.abs(Qp - pQp)).item()
        if max_err < eps:
            # print("Exiting in iteration {}".format(it))
            break

        for t in range(k):
            diff = (-Qp[:, [t]] + pQp) / Q[:, [t]][:, :, [t]]
            p[:, [t]] += diff
            pQp = (pQp + diff * (diff * Q[:, [t]][:, :, [t]] + 2 * Qp[:, [t]])) / (1 + diff) / (1 + diff)
            Qp = (Qp + diff * Q[:, :, [t]]) / (1 + diff)
            p = p / (1 + diff)

    return p.squeeze(2)


def bc(PP, verbose=False):
    device = PP.device
    dtype = PP.dtype
    n, k, kk = PP.size()
    assert k == kk
    eps = 1e-5

    MMi = (1/k)*(torch.eye(k - 1, device=device, dtype=dtype) + torch.ones(k - 1, k - 1, device=device, dtype=dtype))
    rws = int(k * (k - 1) / 2)
    # Mapping h is used such that elements of {1, ..., k(k-1)/2}
    # are placed into upper triangle of k x k matrix row by row from left to right.
    M = torch.zeros(rws, k - 1, device=device, dtype=dtype)
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

    MMiM = torch.matmul(MMi, M.T)

    rv = PP[:, torch.triu(torch.ones(k, k, device=device, dtype=dtype), 1) == 1].T
    small = eps*torch.ones(rv.size(), device=device, dtype=dtype)
    rv = torch.where(rv < eps, small, rv)
    rv = torch.where(rv > 1 - eps, 1 - small, rv)
    s = torch.log(1 / rv - 1)
    u = torch.matmul(MMiM, s)
    zs = torch.zeros(1, n, device=device, dtype=dtype)
    u_exp = torch.exp(torch.cat([zs, u], dim=0))
    ps = u_exp / torch.sum(u_exp, dim=0)

    return ps.T
