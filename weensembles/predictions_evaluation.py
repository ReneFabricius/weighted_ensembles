import torch


@torch.no_grad()
def compute_pairwise_accuracies(MP, tar):
    """
    Computes pairwise accuracies of multiclass classifier

    :param MP: multiclass classifier outputs
    :param tar: correct labels
    :return: vector of pairwise accuracies
    """
    n, k = MP.size()
    pw_acc = torch.zeros((k*(k-1))//2)
    pi = 0
    for fc in range(k):
        for sc in range(fc + 1, k):
            # Obtains fc and sc probabilities for samples belonging to those classes
            SS = MP[(tar == fc) + (tar == sc)][:, [fc, sc]]
            y = tar[(tar == fc) + (tar == sc)]
            mask_fc = (y == fc)
            mask_sc = (y == sc)
            y[mask_fc] = 1
            y[mask_sc] = 0

            top_v, top_i = torch.topk(SS, 1, dim=1)
            ti = top_i.squeeze(1)
            # Coding of target is switched. 1 for class on index 0 and 0 for class on index 1
            pwacc = torch.sum(ti != y) / float(y.size()[0])

            pw_acc[pi] = pwacc

            pi += 1

    return pw_acc


@torch.no_grad()
def get_correctness_masks(MP, tar, topk):
    if type(topk) == int:
        topk = [topk]
    max_k = max(topk)
    top_v, top_i = torch.topk(MP, max_k, dim=1)
    masks = []
    tar_uq = tar.unsqueeze(1)
    for tk in topk:
        mask = torch.sum(top_i[:, :tk] == tar_uq, dim=1)
        if len(topk) == 1:
            return mask

        masks.append(mask.unsqueeze(0))

    return torch.cat(masks, 0)


@torch.no_grad()
def compute_acc_topk(y_cor, ps, l):
    top_v, top_i = torch.topk(ps, l, dim=1)
    n = y_cor.size()[0]

    return torch.sum(top_i == y_cor.unsqueeze(1)).item() / n


@torch.no_grad()
def compute_nll(y_cor, ps, penultimate=False):
    if penultimate:
        lsf = torch.nn.LogSoftmax(dim=1)
        ps_thr = lsf(ps)
    else:
        # min_prob = (1e-16 if ps.dtype == torch.float64 else 1e-7)
        min_prob = 1e-7
        thr = torch.nn.Threshold(min_prob, min_prob)
        ps_thr = thr(ps)
        ps_thr.log_()

    nll = torch.nn.NLLLoss(reduction='sum')
    return nll(ps_thr, y_cor).item()


@torch.no_grad()
def _comp_ece(bin_n, bins, top_probs, cor_pred, p_norm=2):
    ece = 0.0
    monotonic = True
    last_ym = -1
    dtp = top_probs.dtype
    for i in range(bin_n):
        cur = (bins == i)
        if any(cur):
            fxm = torch.mean(top_probs[cur])
            ym = torch.mean(cor_pred[cur].to(dtype=dtp))
            if ym < last_ym:
                monotonic = False
            last_ym = ym
            bin_sam_n = torch.sum(cur)
            ece += bin_sam_n * torch.pow(torch.abs(ym - fxm), p_norm)
    return (torch.pow(ece / top_probs.shape[0], 1. / p_norm)).item(), monotonic


@torch.no_grad()
def ECE_sweep(prob_pred, tar, p_norm=2, penultimate=False):
    """
    Computes estimate of calibration error according to equal mass monotonic sweep algorithm.
    As per https://arxiv.org/abs/2012.08668v2
    :param prob_pred: Probabilities prediction. n×k tensor with n samples and k classes.
    :param tar: Correct labels. n tensor with n samples.
    :param penultimate: If True, applies softmax before computations.
    :return: Estimate of calibration error.
    """
    n, k = prob_pred.shape
    dev = prob_pred.device
    if penultimate:
        prob_pred = torch.nn.Softmax(dim=1)(prob_pred)
    top_probs, top_inds = torch.topk(input=prob_pred, k=1, dim=1)
    top_probs = top_probs.squeeze()
    top_inds = top_inds.squeeze()
    cor_pred = top_inds == tar
    bins = torch.zeros(n, device=dev, dtype=torch.long)
    sample_sort_idxs = torch.argsort(top_probs)
    last_ece = None

    for bin_n in range(2, n + 1):
        bin_assign = torch.min(torch.tensor([bin_n - 1], device=dev, dtype=torch.long),
                               torch.div(torch.arange(start=0, end=n, device=dev) * bin_n, n, rounding_mode='floor').to(dtype=torch.long))
        bins[sample_sort_idxs] = bin_assign
        ece, mon = _comp_ece(bin_n=bin_n, bins=bins, top_probs=top_probs, cor_pred=cor_pred, p_norm=p_norm)
        if mon:
            last_ece = ece
        else:
            if last_ece is not None:
                return last_ece
            else:
                bins = torch.zeros(n, device=dev)
                ece, mon = _comp_ece(bin_n=1, bins=bins, top_probs=top_probs, cor_pred=cor_pred, p_norm=p_norm)
                return ece

    return last_ece
