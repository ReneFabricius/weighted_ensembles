import torch
from timeit import default_timer as timer


def m1(PP, verbose=0):
    """
    Method one of Wu, Lin and Weng.

    :param PP: n×k×k tensor of matrices of pairwise probabilities
    :param verbose: print detailed output
    :return: n×k tensor of probability vectors
    """
    start = timer()
    device = PP.device
    dtype = PP.dtype
    n, k, kk = PP.size()
    assert k == kk
    PP = PP * (1 - torch.eye(k, k, device=device))
    if verbose > 1:
        print("Working with {} samples, each with {} classes".format(n, k))
        if verbose > 2:
            print("Solving for pairwise probabilities\n{}".format(PP.cpu().numpy()))

    E = torch.eye(k, device=device, dtype=dtype)
    Es = E.unsqueeze(0).expand(n, k, k)
    B = torch.zeros(n, k, 1, device=device, dtype=dtype)
    B[:, k - 1, :] = 1

    A = (PP.sum(dim=2).diag_embed() + PP) / (k - 1) - Es
    A[:, k - 1, :] = 1

    if verbose > 2:
        print("Solving linear system\n{}\n× x =\n{}".format(A.cpu().numpy(), B.cpu().numpy()))

    Xs = torch.linalg.solve(A, B)
    
    ps = Xs[:, 0:k, 0:1].squeeze(2)

    if verbose > 2:
        print("Resulting probabilities\n{}".format(ps.cpu().numpy()))

    end = timer()
    if verbose > 0:
        print("Method m1 finished in {:.4f} s".format(end - start))

    return ps


def m2(PP, verbose=0):
    """
    Method two of Wu, Lin and Weng.

    :param PP: n×k×k tensor of matrices of pairwise probabilities
    :param verbose: print detailed output
    :return: n×k tensor of probability vectors
    """
    start = timer()
    device = PP.device
    dtype = PP.dtype
    n, k, kk = PP.size()
    assert k == kk

    PP = PP * (1 - torch.eye(k, k, device=device))
    es = torch.ones(n, k, 1, dtype=dtype, device=device)
    zs = torch.zeros(n, 1, 1, device=device, dtype=dtype)
    B = torch.zeros(n, k + 1, 1, device=device, dtype=dtype)
    B[:, k, :] = 1

    Q = (PP * PP).sum(dim=1).diag_embed() - PP * PP.transpose(1, 2)
    if verbose > 2:
        print("Matrix Q:\n{}".format(Q.cpu().numpy()))

    A = torch.cat((Q, es), 2)
    A = torch.cat((A, torch.cat((es.transpose(1, 2), zs), 2)), 1)

    if verbose > 2:
        print("Solving linear system\n{}\n× x =\n{}".format(A.cpu().numpy(), B.cpu().numpy()))

    Xs = torch.linalg.solve(A, B)
    ps = Xs[:, 0:k, 0:1].squeeze(2)

    if verbose > 2:
        print("Resulting probabilities\n{}".format(ps.cpu().numpy()))

    end = timer()
    if verbose > 0:
        print("Method m2 finished in {:.4f} s".format(end - start))

    return ps


def m2_iter(PP, verbose=0):
    """
    Method two, iterative implementation of Wu, Lin and Weng.

    :param PP: n×k×k tensor of matrices of pairwise probabilities
    :param verbose: print detailed output
    :return: n×k tensor of probability vectors
    """
    start = timer()
    device = PP.device
    dtype = PP.dtype
    n, k, kk = PP.size()
    assert k == kk

    PP = PP * (1 - torch.eye(k, k, device=device))
    max_iter = max(100, k)  # As per LIBSVM implementation
    eps = 1e-12 / k
    min_prob = (1e-16 if dtype == torch.float64 else 1e-7)  # As per LIBSVM

    # Masked select is bugged, uses too much memory and causes memory leak when it fails
    not_diag = torch.logical_xor(torch.ones(k, k, dtype=torch.bool, device=torch.device("cpu")),
                                 torch.eye(k, dtype=torch.bool, device=torch.device("cpu")))

    PP = PP.cpu()
    PP[(PP < min_prob) & not_diag] = min_prob
    PP[(PP > (1 - min_prob)) & not_diag] = 1 - min_prob
    PP = PP.to(device=device)

    Q = (PP * PP).sum(dim=1).diag_embed() - PP * PP.transpose(1, 2)

    if verbose > 2:
        print("Matrix Q:\n{}".format(Q.cpu().numpy()))

    p = torch.ones(n, k, 1, device=device, dtype=dtype) / k

    for it in range(max_iter):
        Qp = torch.matmul(Q, p)
        pQp = torch.matmul(p.transpose(1, 2), Qp)

        max_err = torch.max(torch.abs(Qp - pQp)).item()

        if verbose > 2:
            print("Iteration {}".format(it))
            print("Probability vector:\n{}".format(p.cpu().numpy()))
            print("Qp:\n{}\npQp: {}".format(Qp.cpu().numpy(), pQp.cpu().numpy()))
            print("Maximum error: {}".format(max_err))

        if max_err < eps:
            if verbose > 1:
                print("Exiting in iteration {}".format(it))
            break

        for t in range(k):
            diff = (-Qp[:, [t]] + pQp) / Q[:, [t]][:, :, [t]]
            p[:, [t]] += diff
            pQp = (pQp + diff * (diff * Q[:, [t]][:, :, [t]] + 2 * Qp[:, [t]])) / (1 + diff) / (1 + diff)
            Qp = (Qp + diff * Q[:, :, [t]]) / (1 + diff)
            p = p / (1 + diff)

    end = timer()
    if verbose > 0:
        print("Method m2_iter finished in {:.4f} s".format(end - start))

    return p.squeeze(2)


def bc(PP, verbose=0):
    """
    Bayes covariant method of Such and Barreda.

    :param PP: n×k×k tensor of matrices of pairwise probabilities
    :param verbose: print detailed output
    :return: n×k tensor of probability vectors
    """
    start = timer()
    device = PP.device
    dtype = PP.dtype
    n, k, kk = PP.size()
    assert k == kk
    eps = 1e-5

    PP = PP * (1 - torch.eye(k, k, device=device))
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

    if verbose > 2:
        print("Matrix MMiM:\n{}".format(MMiM.cpu().numpy()))

    rv = PP[:, torch.triu(torch.ones(k, k, device=device, dtype=dtype), 1) == 1].T
    small = eps*torch.ones(rv.size(), device=device, dtype=dtype)
    rv = torch.where(rv < eps, small, rv)
    rv = torch.where(rv > 1 - eps, 1 - small, rv)
    s = torch.log(1 / rv - 1)
    u = torch.matmul(MMiM, s)

    if verbose > 2:
        print("Vector s:\n{}\nVector u:\n{}".format(s.cpu().numpy(), u.cpu().numpy()))

    zs = torch.zeros(1, n, device=device, dtype=dtype)
    u_exp = torch.exp(torch.cat([zs, u], dim=0))
    ps = u_exp / torch.sum(u_exp, dim=0)

    end = timer()
    if verbose > 0:    
        print("Method bc finished in {:.4f} s".format(end - start))

    return ps.T


def sbt(PP, verbose=0):
    """
    Coupling method of Such, Benus and Tinajova.

    :param PP: n×k×k tensor of matrices of pairwise probabilities
    :param verbose: print detailed output
    :return: n×k tensor of probability vectors
    """
    
    dtp = PP.dtype
    dev = PP.device

    start = timer()
    n, k, kk = PP.shape
    assert(k == kk)
    PP = PP * (1 - torch.eye(k, k, device=dev))
    if verbose > 1:
        print("Working with {} samples, each with {} classes".format(n, k))
        if verbose > 2:
            print("Solving for pairwise probabilities\n{}".format(PP.cpu().numpy()))

    ey = torch.eye(k, dtype=dtp, device=dev)
    R_zd = PP * (1 - ey)
    ND = torch.div(1, ey + R_zd.transpose(dim0=-2, dim1=-1)) - 1 + ey
    DI = torch.diag_embed(torch.div(1, torch.sum(torch.div(1, R_zd + ey), dim=-1) - (k - 1)))
    P = torch.matmul(ND, DI) - ey
    P[:, k - 1, :] = 1
    B = torch.zeros(n, k, 1, dtype=dtp, device=dev)
    B[:, k - 1, :] = 1
    if verbose > 2:
        print("Solving linear system\n{}\n× x =\n{}".format(P.cpu().numpy(), B.cpu().numpy()))

    X = torch.linalg.lstsq(P, B).solution
    
    end = timer()
    if verbose > 2:
        print("Resulting probabilities\n{}".format(X.squeeze(2).cpu().numpy()))

    end = timer()
    if verbose > 0:
        print("Method sbt finished in {:.4f} s".format(end - start))

    return X.squeeze(2)
        

coup_methods = {"m1": m1,
                "m2": m2,
                "bc": bc,
                "m2_iter": m2_iter,
                "sbt": sbt}

def coup_picker(cp_m):
    if cp_m not in coup_methods:
        return None 
    
    return coup_methods[cp_m]
