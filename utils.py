import torch
from my_codes.weighted_ensembles.WeightedLDAEnsemble import logit_sparse, logit
from my_codes.weighted_ensembles.SimplePWCombine import bc


def gen_probs(c, n, k):
    p = torch.randn(c, n, k).cuda()
    p = p - torch.min(p)
    p = p / torch.sum(p, 2).unsqueeze(2)
    return p


def pwp(MP, l):
    dev = MP.device
    c, n, k = MP.size()

    coefs = torch.rand(k, k, c + 1, device=dev)

    val, ind = torch.topk(MP, l, dim=2)
    M = torch.zeros(MP.shape, dtype=torch.bool, device=dev)
    # place true in positions of top probs
    M.scatter_(2, ind, True)
    # combine selections over c inputs
    M = torch.sum(M, dim=0, dtype=torch.bool)
    # zeroe lower values
    MPz = MP*M
    MPz.transpose_(0, 1)
    ps = torch.zeros(n, k, device=dev)
    # Selected class counts for every n
    NPC = torch.sum(M, 1).squeeze()
    for pc in range(l, l * c + 1):
        # Pick pc-class samples
        pcMPz = MPz[NPC == pc]
        # Pick pc-class masks
        pcM = M[NPC == pc]
        pcn = pcM.shape[0]
        if pcn == 0:
            continue

        pcIMR = torch.tensor([], dtype=torch.long, device=dev)
        pcIMC = torch.tensor([], dtype=torch.long, device=dev)
        for r in range(pc):
            pcIMR = torch.cat((pcIMR, torch.tensor([r], dtype=torch.long, device=dev).repeat(pc - r - 1)))
            pcIMC = torch.cat((pcIMC, torch.arange(r + 1, pc, device=dev)))

        # Indexes of picked values
        pcMi = torch.nonzero(pcM, as_tuple=False)[:, 1].view(pcM.shape[0], pc)
        # Just picked values
        pcMPp = pcMPz.gather(2, pcMi.unsqueeze(1).expand(pcn, c, pc))

        # For every pcn and c contains values of top right triangle of ps expanded as columns
        pcMPpR = pcMPp[:, :, pcIMR]
        # For every pcn and c contains values of top right triangle of ps expanded as rows
        pcMPpC = pcMPp[:, :, pcIMC]
        # For every pcn and c contains top right triangle of pairwise probs p_ij
        pcPWP = pcMPpR / (pcMPpR + pcMPpC)

        # logit pairwise probs
        pcLI = logit(pcPWP, 1e-5)
        # Flattened logits in order of dimensions: pcn; kxk top right triangle by rows; c
        pcLIflat = pcLI.transpose(1, 2).flatten()

        # Number of values in top right triangle
        val_ps = pc * (pc - 1) // 2
        # kxk matrix row indexes of values in pcLIflat without considering c sources
        I1woc = pcMi[:, pcIMR]
        # kxk matrix row indexes of values in pcLIflat
        I1 = I1woc.repeat_interleave(c)
        # kxk matrix column indexes of values in pcLIflat without considering c sources
        I2woc = pcMi[:, pcIMC]
        # kxk matrix column indexes of values in pcLIflat
        I2 = I2woc.repeat_interleave(c)
        # source indexes of values in pcLIflat
        I3 = torch.arange(c, device=dev).repeat(pcn * val_ps)

        # Extract lda coefficients
        Ws = coefs[I1, I2, I3]
        Bs = coefs[I1woc.flatten(), I2woc.flatten(), c]

        # Apply lda predict_proba
        pcLC = pcLIflat * Ws
        pcDEC = torch.sum(pcLC.view(pcn*val_ps, c), 1) + Bs
        CPWP = 1 / (1 + torch.exp(-pcDEC))

        # Build dense matrices of pairwise probabilities disregarding original positions in all-class setting
        dI0 = torch.arange(0, pcn, device=dev).repeat_interleave(val_ps)
        dI1 = pcIMR.repeat(pcn)
        dI2 = pcIMC.repeat(pcn)

        I = torch.cat((dI0.unsqueeze(0), dI1.unsqueeze(0), dI2.unsqueeze(0)), 0)
        DPS = torch.sparse_coo_tensor(I, CPWP, (pcn, pc, pc), device=dev).to_dense()
        It = torch.cat((dI0.unsqueeze(0), dI2.unsqueeze(0), dI1.unsqueeze(0)), 0)
        DPSt = torch.sparse_coo_tensor(It, 1 - CPWP, (pcn, pc, pc), device=dev).to_dense()
        # DPS now should contain pairwise probabilities
        DPS = DPS + DPSt

        pcPS = bc(DPS)

        # resulting posteriors for samples with pc picked classes
        ps_cur = torch.zeros(pcn, k, device=dev)
        row_mask = (NPC == pc)
        ps_cur[M[row_mask]] = torch.flatten(pcPS)
        # Insert current results into complete tensor of posteriors
        ps[row_mask] = ps_cur

    return ps
