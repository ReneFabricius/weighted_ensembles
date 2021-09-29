import torch
from my_codes.weighted_ensembles.WeightedLDAEnsemble import pairwise_accuracies, logit


def compute_pairwise_accuracies(MP, tar):
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


def compute_acc_topk(y_cor, ps, l):
    top_v, top_i = torch.topk(ps, l, dim=1)
    n = y_cor.size()[0]

    return torch.sum(top_i == y_cor.unsqueeze(1)).item() / n


def compute_nll(y_cor, ps):
    min_prob = (1e-16 if ps.dtype == torch.float64 else 1e-7)
    thr = torch.nn.Threshold(min_prob, min_prob)
    ps_thr = thr(ps)
    ps_thr.log_()
    nll = torch.nn.NLLLoss(reduction='sum')
    return nll(ps_thr, y_cor)
