import torch


def gen_probs(c, n, k, device=torch.device("cuda"), dtype=torch.float32):
    """
    Generates random outputs of probabilistic classifiers.

    :param c: number of classifiers
    :param n: number of samples
    :param k: number of classes
    :param device: device on which to output the result
    :param dtype: dtype in which to output the result
    :return: c×n×k tensor of probabilities with last dimension summing to 1
    """
    p = torch.randn(c, n, k, device=device, dtype=dtype)
    p = p - torch.min(p)
    p = p / torch.sum(p, 2).unsqueeze(2)
    return p


def gen_probs_one_source(n, k, device=torch.device("cuda"), dtype=torch.float32):
    """
    Generates random outputs of probabilistic classifier.

    :param n: number of samples
    :param k: number of classes
    :param device: device on which to output the result
    :param dtype: dtype in which to return the result
    :return: n×k tensor of probabilities with last dimension summing to 1
    """
    p = gen_probs(1, n, k, device=device, dtype=dtype)
    p = p.squeeze(dim=0)
    return p


def comp_R(p):
    """
    Computes matrices of pairwise probabilities from classifier predictions
    :param p: classifier predictions
    :return: matrices of pairwise probabilities
    """
    dev = p.device
    dtp = p.dtype
    if len(p.shape) == 2:
        n, k = p.shape
        E = torch.eye(k, device=dev, dtype=dtp)
        VE = (1 - E).unsqueeze(0).expand(n, k, k)
        TCs = VE * p.unsqueeze(2)
    else:
        c, n, k = p.shape
        E = torch.eye(k, device=dev, dtype=dtp)
        VE = (1 - E).unsqueeze(0).unsqueeze(1).expand(c, n, k, k)
        TCs = VE * p.unsqueeze(3)

    R = TCs / (TCs + TCs.transpose(-1, -2) + (TCs == 0))
    return R


def logit(T, eps):
    """
    Computes logit of the input.
    Replaces values below eps by eps and above 1 - eps by 1 - eps before computing the logit.

    :param T: input
    :param eps: epsilon
    :return: logit of the input
    """
    EPS = T.new_full(T.shape, eps, device=T.device, dtype=T.dtype)
    L = torch.where(T < eps, EPS, T)
    LU = torch.where(L > 1 - eps, 1 - EPS, L)
    return torch.log(LU/(1 - LU))


def logit_sparse(T, eps):
    """
    Logit function for sparse tensors.
    Computes logit of the input.
    Replaces values below eps by eps and above 1 - eps by 1 - eps before computing the logit.

    :param T: sparse input
    :param eps: epsilon
    :return: sparse logit of the input
    """
    Vli = logit(T.values(), eps)
    return torch.sparse_coo_tensor(T.indices(), Vli, T.shape)


def pairwise_accuracies(SS, tar):
    """
    Computes accuracy of binary probabilistic classifiers with switched class labels.

    :param SS: c×n×2 tensor, where c is number of classifiers, n is number of samples
    :param tar: correct labels, 0 for class on index 1 and 1 for class on index 0
    :return: accuracies of classifiers
    """
    c, n, k = SS.size()
    top_v, top_i = torch.topk(SS, 1, dim=2)
    ti = top_i.squeeze(dim=2)
    # Coding of target is switched. 1 for class on index 0 and 0 for class on index 1
    return torch.sum(ti.cpu() != tar, dim=1)/float(n)


def pairwise_accuracies_penultimate(SS, tar):
    """
        Computes accuracy of difference of supports with switched class labels.

        :param SS: c×n tensor, where c is number of classifiers, n is number of samples,
        positive value means class one, negative class two
        :param tar: correct labels, 1 means class one, 0 means class two
        :return: accuracies of classifiers
        """
    c, n = SS.size()
    ti = SS > 0
    return torch.sum(ti == tar, dim=1) / float(n)


def cuda_mem_try(fun, start_bsz, device, dec_coef=0.5, max_tries=None, verbose=0):
    """Repeatedly to perform action specified by given function which could fail due to cuda oom.
    Each try is performed with lower batch size than previous.

    Args:
        fun (function): Function with one argument - batch size.
        start_bsz (int): STarting batch size.
        dec_coef (float, optional): Coeficient used to multiplicatively decrease batch size. Defaults to 0.5.
        max_tries (int, optional): Maximum number of tries. If None, tries are not limited. Defaults to None.
    """
    batch_size = start_bsz
    while batch_size > 0 and (max_tries is None or max_tries > 0):
        try:
            if verbose > 1:
                print("Trying with batch size {}".format(batch_size))
            return fun(batch_size)
        except RuntimeError as rerr:
            str_err = str(rerr)
            if "memory" not in str_err and "CUDA" not in str_err: 
                raise rerr
            if verbose > 1:
                print("CUDA oom exception")
            del rerr
            batch_size = int(dec_coef * batch_size)
            if max_tries is not None:
                max_tries -= 1
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    else:
        raise RuntimeError("Unsuccessful to perform the requested action. CUDA out of memory.")
            
            