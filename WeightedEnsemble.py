import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import pandas as pd
from statsmodels.stats.diagnostic import normal_ad

from timeit import default_timer as timer


def logit(T, eps):
    EPS = T.new_full(T.shape, eps, device=T.device)
    L = torch.where(T < eps, EPS, T)
    LU = torch.where(L > 1 - eps, 1 - EPS, L)
    return torch.log(LU/(1 - LU))


def logit_sparse(T, eps):
    Vli = logit(T.values(), eps)
    return torch.sparse_coo_tensor(T.indices(), Vli, T.shape)


def pairwise_accuracies(SS, tar):
    c, n, k = SS.size()
    top_v, top_i = torch.topk(SS, 1, dim=2)
    ti = top_i.squeeze(dim=2)
    # Coding of target is switched. 1 for class on index 0 and 0 for class on index 1
    return torch.sum(ti.cpu() != tar, dim=1)/float(n)


class WeightedEnsemble:
    def __init__(self, c=0, k=0, device=torch.device("cpu")):
        """

        :param c: Number of classifiers
        :param k: Number of classes
        :param device:
        """
        self.dev_ = device
        self.logit_eps_ = 1e-5
        self.c_ = c
        self.k_ = k
        self.coefs_ = torch.zeros(k, k, c + 1).to(self.dev_)
        self.ldas_ = [[None for j in range(k)] for i in range(k)]
        self.pvals_ = None

    def fit(self, MP, tar, verbose=False, test_normality=False):
        """Trains lda for every pair of classes"""
        print("Starting fit")
        start = timer()
        num = self.k_*(self.k_ - 1)//2      # Number of pairs of classes
        if test_normality:
            self.pvals_ = torch.zeros(num, 2, self.c_).to(torch.device("cpu"))
        print_step = num // 100
        pi = 0
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                if print_step > 0 and pi % print_step == 0:
                    print("Fit progress " + str(pi // print_step) + "%", end="\r")

                # k x s x 2 tensor, where s is number of samples in classes fc and sc.
                # Tensor contains supports of networks for classes fc, sc for samples belonging to fc, sc.
                SS = MP[:, (tar == fc) + (tar == sc)][:, :, [fc, sc]].to(self.dev_)
                # k x s tensor containing p_fc,sc pairwise probabilities for above mentioned samples
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                LI = logit(PWP, self.logit_eps_)
                # k x s tensor of logit supports of k networks for class fc against class sc for s samples
                X = LI.transpose(0, 1).cpu()
                # Prepare targets
                y = tar[(tar == fc) + (tar == sc)]
                mask_fc = (y == fc)
                mask_sc = (y == sc)
                y[mask_fc] = 1
                y[mask_sc] = 0

                if test_normality:
                    # Test normality of predictors
                    fc_pval = torch.tensor([normal_ad(X[y == 1][:, ci].numpy(), 0)[1] for ci in range(self.c_)])
                    sc_pval = torch.tensor([normal_ad(X[y == 0][:, ci].numpy(), 0)[1] for ci in range(self.c_)])
                    self.pvals_[pi, 0, :] = fc_pval
                    self.pvals_[pi, 1, :] = sc_pval
                    if verbose:
                        print("P-values of normality test for class " + str(fc))
                        print(str(fc_pval))
                        print("P-values of normality test for class " + str(sc))
                        print(str(sc_pval))

                clf = LinearDiscriminantAnalysis(solver='lsqr')
                clf.fit(X, y)
                self.ldas_[fc][sc] = clf
                self.coefs_[fc, sc, :] = torch.cat((torch.tensor(clf.coef_).to(self.dev_).squeeze(),
                                                    torch.tensor(clf.intercept_).to(self.dev_)))

                if verbose:
                    pwacc = pairwise_accuracies(SS, y)
                    print("Training pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                          "\n\tpairwise accuracies: " + str(pwacc) +
                          "\n\tchosen coefficients: " + str(clf.coef_) +
                          "\n\tintercept: " + str(clf.intercept_))

                    print("\tcombined accuracy: " + str(clf.score(X, y)))

                pi += 1

        if test_normality and verbose:
            blw_5 = torch.sum(self.pvals_ < 0.05)
            blw_1 = torch.sum(self.pvals_ < 0.01)
            print("Number of classes with pval below 5% " + str(blw_5.item()))
            print("Number of classes with pval below 1% " + str(blw_1.item()))

        end = timer()
        print("Fit finished in " + str(end - start) + " s")

    def predict_proba(self, MP, PWComb):
        print("Starting predict proba")
        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_
        p_probs = torch.Tensor(n, k, k)

        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                # Obtains fc and sc probabilities for all classes
                SS = MP[:, :, [fc, sc]].to(self.dev_)
                # Computes p_ij pairwise probabilities for above mentioned samples
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                LI = logit(PWP, self.logit_eps_)
                X = LI.transpose(0, 1).cpu()
                PP = self.ldas_[fc][sc].predict_proba(X)
                p_probs[:, sc, fc] = torch.from_numpy(PP[:, 0])
                p_probs[:, fc, sc] = torch.from_numpy(PP[:, 1])

        end = timer()
        print("Predict proba finished in " + str(end - start) + " s")

        return PWComb(p_probs.to(self.dev_))

    def test_pairwise(self, MP, tar):
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                # Obtains fc and sc probabilities for samples belonging to those classes
                SS = MP[:, (tar == fc) + (tar == sc)][:, :, [fc, sc]].to(self.dev_)
                # Computes p_ij pairwise probabilities for above mentioned samples
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                LI = logit(PWP, self.logit_eps_)
                X = LI.transpose(0, 1).cpu()
                y = tar[(tar == fc) + (tar == sc)]
                mask_fc = (y == fc)
                mask_sc = (y == sc)
                y[mask_fc] = 1
                y[mask_sc] = 0

                pwacc = pairwise_accuracies(SS, y)
                print(
                    "Testing pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                    "\n\tpairwise accuracies: " + str(pwacc) +
                    "\n\tchosen coefficients: " + str(self.ldas_[fc][sc].coef_) +
                    "\n\tintercept: " + str(self.ldas_[fc][sc].intercept_))

                print("\tcombined accuracy: " + str(self.ldas_[fc][sc].score(X, y)))

    def predict_proba_topl(self, MP, l, PWComb):
        print("Starting predict proba topl")
        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_

        ps = torch.zeros(n, k).to(self.dev_)
        # Every sample may have different set of top classes, so we process them one by one
        print_step = n // 100
        for ni in range(n):
            if ni % print_step == 0:
                print("Predicting proba topl progress " + str(ni//print_step) + "%", end="\r")

            val, ind = torch.topk(MP[:, ni, :], l, dim=1)
            Ti = sorted(list(set(ind.flatten().tolist())))
            tcc = len(Ti)
            p_probs = torch.zeros(1, tcc, tcc).to(self.dev_)
            for fci, fc in enumerate(Ti):
                for sci, sc in enumerate(Ti[fci + 1:]):
                    # Obtains fc and sc probabilities for current sample
                    SS = MP[:, ni, [fc, sc]].to(self.dev_)
                    # Computes p_ij pairwise probabilities for above mentioned samples
                    PWP = torch.true_divide(SS[:, 0], torch.sum(SS, 1) + (SS[:, 0] == 0))
                    LI = logit(PWP, self.logit_eps_)
                    X = LI.T.unsqueeze(0).cpu()
                    PP = self.ldas_[fc][sc].predict_proba(X)
                    p_probs[:, fci + 1 + sci, fci] = torch.from_numpy(PP[:, 0])
                    p_probs[:, fci, fci + 1 + sci] = torch.from_numpy(PP[:, 1])

            sam_ps = PWComb(p_probs.to(self.dev_))
            ps[ni, Ti] = sam_ps.squeeze()

        end = timer()
        print("Predict proba topl finished in " + str(end - start) + " s")

        return ps

    def predict_proba_topl_fast(self, MP, l, PWComb):
        """Should produce same results as predict_proba_topl"""
        print("Starting predict proba topl fast")
        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_

        MP = MP.to(self.dev_)
        val, ind = torch.topk(MP, l, dim=2)
        M = torch.zeros(MP.shape, dtype=torch.bool, device=self.dev_)
        # place true in positions of top probs
        M.scatter_(2, ind, True)
        # combine selections over c inputs
        M = torch.sum(M, dim=0, dtype=torch.bool)
        # zeroe lower values
        MPz = MP * M
        MPz.transpose_(0, 1)
        ps = torch.zeros(n, k, device=self.dev_)
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

            pcIMR = torch.tensor([], dtype=torch.long, device=self.dev_)
            pcIMC = torch.tensor([], dtype=torch.long, device=self.dev_)
            for r in range(pc):
                pcIMR = torch.cat((pcIMR, torch.tensor([r], dtype=torch.long, device=self.dev_).repeat(pc - r - 1)))
                pcIMC = torch.cat((pcIMC, torch.arange(r + 1, pc, device=self.dev_)))

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
            I3 = torch.arange(c, device=self.dev_).repeat(pcn * val_ps)

            # Extract lda coefficients
            Ws = self.coefs_[I1, I2, I3]
            Bs = self.coefs_[I1woc.flatten(), I2woc.flatten(), c]

            # Apply lda predict_proba
            pcLC = pcLIflat * Ws
            pcDEC = torch.sum(pcLC.view(pcn * val_ps, c), 1) + Bs
            CPWP = 1 / (1 + torch.exp(-pcDEC))

            # Build dense matrices of pairwise probabilities disregarding original positions in all-class setting
            dI0 = torch.arange(0, pcn, device=self.dev_).repeat_interleave(val_ps)
            dI1 = pcIMR.repeat(pcn)
            dI2 = pcIMC.repeat(pcn)

            I = torch.cat((dI0.unsqueeze(0), dI1.unsqueeze(0), dI2.unsqueeze(0)), 0)
            DPS = torch.sparse_coo_tensor(I, CPWP, (pcn, pc, pc), device=self.dev_).to_dense()
            It = torch.cat((dI0.unsqueeze(0), dI2.unsqueeze(0), dI1.unsqueeze(0)), 0)
            DPSt = torch.sparse_coo_tensor(It, 1 - CPWP, (pcn, pc, pc), device=self.dev_).to_dense()
            # DPS now should contain pairwise probabilities
            DPS = DPS + DPSt

            pcPS = PWComb(DPS)

            # resulting posteriors for samples with pc picked classes
            ps_cur = torch.zeros(pcn, k, device=self.dev_)
            row_mask = (NPC == pc)
            ps_cur[M[row_mask]] = torch.flatten(pcPS)
            # Insert current results into complete tensor of posteriors
            ps[row_mask] = ps_cur

        end = timer()
        print("Predict proba topl fast finished in " + str(end - start) + " s")

        return ps

    def save_models(self, file):
        print("Saving models into file: " + str(file))
        with open(file, 'wb') as f:
            pickle.dump(self.ldas_, f)

    def load_models(self, file):
        print("Loading models from file: " + str(file))
        with open(file, 'rb') as f:
            self.ldas_ = pickle.load(f)

        self.k_ = len(self.ldas_)
        if self.k_ > 0:
            self.c_ = len(self.ldas_[0][1].coef_[0])

        self.coefs_ = torch.zeros(self.k_, self.k_, self.c_ + 1).to(self.dev_)

        for fc in range(len(self.ldas_)):
            for sc in range(fc + 1, len(self.ldas_[fc])):
                clf = self.ldas_[fc][sc]
                self.coefs_[fc, sc, :] = torch.cat((torch.tensor(clf.coef_).to(self.dev_).squeeze(),
                                                    torch.tensor(clf.intercept_).to(self.dev_)))

    def save_coefs_csv(self, file):
        Ls = [None] * ((self.k_ * (self.k_ - 1)) // 2)
        li = 0
        cols = ["i", "j"] + ["coef" + str(k) for k in range(self.c_)] + ["interc"]
        for i in range(self.k_):
            for j in range(i + 1, self.k_):
                cfs = [[i, j] + self.coefs_[i, j].tolist()]
                Ls[li] = pd.DataFrame(cfs, columns=cols)
                li += 1

        df = pd.concat(Ls, ignore_index=True)
        df.to_csv(file, index=False)

    def save_pvals(self, file):
        if self.pvals_ is not None:
            np.save(file, self.pvals_)
        else:
            print("P-values not computed")


