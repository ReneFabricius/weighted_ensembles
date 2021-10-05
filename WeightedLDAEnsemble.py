import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import pandas as pd
from scipy.stats import normaltest

from timeit import default_timer as timer


def logit(T, eps):
    EPS = T.new_full(T.shape, eps, device=T.device, dtype=T.dtype)
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


def pairwise_accuracies_penultimate(SS, tar):
    c, n = SS.size()
    ti = SS > 0
    return torch.sum(ti == tar, dim=1) / float(n)


class WeightedLDAEnsemble:
    def __init__(self, c=0, k=0, device=torch.device("cpu"), dtp=torch.float32):
        """
        Trainable ensembling of classification posteriors.
        :param c: Number of classifiers
        :param k: Number of classes
        :param device: Torch device to use for computations
        :param dtp: Torch datatype to use for computations
        """
        self.dev_ = device
        self.dtp_ = dtp
        self.logit_eps_ = 1e-5
        self.c_ = c
        self.k_ = k
        self.coefs_ = torch.zeros(k, k, c + 1, device=self.dev_, dtype=self.dtp_)
        self.ldas_ = [[None for _ in range(k)] for _ in range(k)]
        self.pvals_ = None
        self.trained_on_penultimate_ = None

    def fit(self, MP, tar, verbose=False, test_normality=False):
        """
        Trains lda on logits of pairwise probabilities for every pair of classes
        :param MP: c x n x k tensor of constituent classifiers outputs.
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param tar: n tensor of sample labels
        :param verbose: print more detailed output
        :param test_normality: test normality of lda predictors for each class in each class pair
        :return:
        """

        print("Starting fit")
        start = timer()
        with torch.no_grad():
            num = self.k_ * (self.k_ - 1) // 2      # Number of pairs of classes
            if test_normality:
                self.pvals_ = torch.zeros(num, 2, self.c_, device=torch.device("cpu"), dtype=self.dtp_)
            print_step = num // 100

            num_non_one = torch.sum(torch.abs(torch.sum(MP, dim=2) - 1.0) > self.logit_eps_).item()
            if num_non_one > 0:
                print("Warning: " + str(num_non_one) +
                      " samples with non unit sum of supports found, performing softmax")
                MP = self.softmax_supports(MP)

            pi = 0
            for fc in range(self.k_):
                for sc in range(fc + 1, self.k_):
                    if print_step > 0 and pi % print_step == 0:
                        print("Fit progress " + str(pi // print_step) + "%", end="\r")

                    # c x s x 2 tensor, where s is number of samples in classes fc and sc.
                    # Tensor contains supports of networks for classes fc, sc for samples belonging to fc, sc.
                    SS = MP[:, (tar == fc) + (tar == sc)][:, :, [fc, sc]].to(device=self.dev_, dtype=self.dtp_)
                    # c x s tensor containing p_fc,sc pairwise probabilities for above mentioned samples
                    PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                    LI = logit(PWP, self.logit_eps_)
                    # s x c tensor of logit supports of k networks for class fc against class sc for s samples
                    X = LI.transpose(0, 1).cpu()
                    # Prepare targets
                    y = tar[(tar == fc) + (tar == sc)]
                    mask_fc = (y == fc)
                    mask_sc = (y == sc)
                    y[mask_fc] = 1
                    y[mask_sc] = 0

                    if test_normality:
                        # Test normality of predictors
                        #fc_pval = torch.tensor([normal_ad(X[mask_fc][:, ci].numpy(), 0)[1] for ci in range(self.c_)])
                        #sc_pval = torch.tensor([normal_ad(X[mask_sc][:, ci].numpy(), 0)[1] for ci in range(self.c_)])
                        fc_pval = torch.tensor(normaltest(X[mask_fc], 0)[1])
                        sc_pval = torch.tensor(normaltest(X[mask_sc], 0)[1])
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
                    self.coefs_[fc, sc, :] = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                                                        torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))

                    if verbose:
                        pwacc = pairwise_accuracies(SS, y)
                        print("Training pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                              "\n\tpairwise accuracies: " + str(pwacc) +
                              "\n\tchosen coefficients: " + str(clf.coef_) +
                              "\n\tintercept: " + str(clf.intercept_))

                        print("\tcombined accuracy: " + str(clf.score(X, y)))

                    pi += 1

            if test_normality:
                blw_5 = torch.sum(self.pvals_ < 0.05)
                blw_1 = torch.sum(self.pvals_ < 0.01)
                print("Number of classes with normality pval below 5% " + str(blw_5.item()))
                print("Number of classes with normality pval below 1% " + str(blw_1.item()))

        end = timer()
        self.trained_on_penultimate_ = False
        print("Fit finished in " + str(end - start) + " s")

    def fit_penultimate(self, MP, tar, verbose=False, test_normality=False):
        """
        Trains lda on supports of penultimate layer for every pair of classes
        :param MP: c x n x k tensor of constituent classifiers penultimate layer outputs.
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param tar: n tensor of sample labels
        :param verbose: print more detailed output
        :param test_normality: test normality of lda predictors for each class in each class pair
        :return:
        """

        print("Starting fit")
        start = timer()
        with torch.no_grad():
            num = self.k_ * (self.k_ - 1) // 2      # Number of pairs of classes
            if test_normality:
                self.pvals_ = torch.zeros(num, 2, self.c_, device=torch.device("cpu"), dtype=self.dtp_)
            print_step = num // 100

            pi = 0
            for fc in range(self.k_):
                for sc in range(fc + 1, self.k_):
                    if print_step > 0 and pi % print_step == 0:
                        print("Fit progress " + str(pi // print_step) + "%", end="\r")

                    # c x n tensor containing True for samples belonging to classes fc, sc
                    SamM = (tar == fc) + (tar == sc)
                    # c x s x 1 tensor, where s is number of samples in classes fc and sc.
                    # Tensor contains support of networks for class fc minus support for class sc
                    SS = MP[:, SamM][:, :, fc] - MP[:, SamM][:, :, sc]

                    # s x c tensor of logit supports of k networks for class fc against class sc for s samples
                    X = SS.squeeze().transpose(0, 1)
                    # Prepare targets
                    y = tar[SamM]
                    mask_fc = (y == fc)
                    mask_sc = (y == sc)
                    y[mask_fc] = 1
                    y[mask_sc] = 0

                    if test_normality:
                        # Test normality of predictors
                        # fc_pval = torch.tensor([normal_ad(X[mask_fc][:, ci].numpy(), 0)[1] for ci in range(self.c_)])
                        # sc_pval = torch.tensor([normal_ad(X[mask_sc][:, ci].numpy(), 0)[1] for ci in range(self.c_)])
                        fc_pval = torch.tensor(normaltest(X[mask_fc].detach().cpu(), 0)[1])
                        sc_pval = torch.tensor(normaltest(X[mask_sc].detach().cpu(), 0)[1])
                        self.pvals_[pi, 0, :] = fc_pval
                        self.pvals_[pi, 1, :] = sc_pval
                        if verbose:
                            print("P-values of normality test for class " + str(fc))
                            print(str(fc_pval))
                            print("P-values of normality test for class " + str(sc))
                            print(str(sc_pval))

                    clf = LinearDiscriminantAnalysis(solver='lsqr')
                    clf.fit(X.detach().cpu(), y.detach().cpu())
                    self.ldas_[fc][sc] = clf
                    self.coefs_[fc, sc, :] = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                                                        torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))

                    if verbose:
                        pwacc = pairwise_accuracies_penultimate(SS, y)
                        print("Training pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                              "\n\tpairwise accuracies: " + str(pwacc) +
                              "\n\tchosen coefficients: " + str(clf.coef_) +
                              "\n\tintercept: " + str(clf.intercept_))
                        if self.dev_.type == "cpu":
                            print("\tcombined accuracy: " + str(clf.score(X, y)))
                        else:
                            print("\tcombined accuracy: " + str(clf.score(X.detach().cpu(), y.detach().cpu())))

                    pi += 1

            if test_normality:
                blw_5 = torch.sum(self.pvals_ < 0.05)
                blw_1 = torch.sum(self.pvals_ < 0.01)
                print("Number of classes with normality pval below 5% " + str(blw_5.item()))
                print("Number of classes with normality pval below 1% " + str(blw_1.item()))

        end = timer()
        self.trained_on_penultimate_ = True
        print("Fit finished in " + str(end - start) + " s")

    '''def fit_fast_penultimate(self, MP, tar, verbose=False):
        """
        Trains lda on supports of penultimate layer for every pair of classes
        :param MP: c x n x k tensor of constituent classifiers penultimate layer outputs.
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param tar: n tensor of sample labels
        :param verbose: print more detailed output
        :param test_normality: test normality of lda predictors for each class in each class pair
        :return:
        """

        print("Starting fit fast")
        start = timer()
        with torch.no_grad():
            tar = tar.to(device=self.dev_, dtype=self.dtp_)
            MP = MP.to(device=self.dev_, dtype=self.dtp_)
            class_counts = torch.bincount(tar)
            priors = class_counts / torch.sum(class_counts)




            pi = 0
            for fc in range(self.k_):
                for sc in range(fc + 1, self.k_):
                    if print_step > 0 and pi % print_step == 0:
                        print("Fit progress " + str(pi // print_step) + "%", end="\r")

                    # c x n tensor containing True for samples belonging to classes fc, sc
                    SamM = (tar == fc) + (tar == sc)
                    # c x s x 1 tensor, where s is number of samples in classes fc and sc.
                    # Tensor contains support of networks for class fc minus support for class sc
                    SS = MP[:, SamM][:, :, fc] - MP[:, SamM][:, :, sc]

                    # s x c tensor of logit supports of k networks for class fc against class sc for s samples
                    X = SS.squeeze().transpose(0, 1)
                    # Prepare targets
                    y = tar[SamM]
                    mask_fc = (y == fc)
                    mask_sc = (y == sc)
                    y[mask_fc] = 1
                    y[mask_sc] = 0

                    clf = LinearDiscriminantAnalysis(solver='lsqr')
                    clf.fit(X.detach().cpu(), y.detach().cpu())
                    self.ldas_[fc][sc] = clf
                    self.coefs_[fc, sc, :] = torch.cat(
                        (torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                         torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))

                    if verbose:
                        pwacc = pairwise_accuracies_penultimate(SS, y)
                        print("Training pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                              "\n\tpairwise accuracies: " + str(pwacc) +
                              "\n\tchosen coefficients: " + str(clf.coef_) +
                              "\n\tintercept: " + str(clf.intercept_))
                        if self.dev_.type == "cpu":
                            print("\tcombined accuracy: " + str(clf.score(X, y)))
                        else:
                            print("\tcombined accuracy: " + str(clf.score(X.detach().cpu(), y.detach().cpu())))

                    pi += 1

        end = timer()
        self.trained_on_penultimate_ = True
        print("Fit finished in " + str(end - start) + " s")'''

    def predict_proba(self, MP, PWComb, debug_pwcm=False):
        """
        Combines outputs of constituent classifiers using all classes.
        :param MP: c x n x k tensor of constituent classifiers posteriors
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param PWComb: coupling method to use
        :return: n x k tensor of combined posteriors
        """
        print("Starting predict proba, pwc method {}".format(PWComb.__name__))
        if self.trained_on_penultimate_ is None:
            print("Ensemble not trained")
            return

        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_
        p_probs = torch.zeros(n, k, k, dtype=self.dtp_)

        if not self.trained_on_penultimate_:
            num_non_one = torch.sum(torch.abs(torch.sum(MP, dim=2) - 1.0) > self.logit_eps_).item()
            if num_non_one > 0:
                print("Warning: " + str(num_non_one) +
                      " samples with non unit sum of supports found, performing softmax")
                MP = self.softmax_supports(MP)

        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                if not self.trained_on_penultimate_:
                    # Obtains fc and sc probabilities for all classes
                    SS = MP[:, :, [fc, sc]].to(device=self.dev_, dtype=self.dtp_)
                    # Computes p_ij pairwise probabilities for above mentioned samples
                    PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                    LI = logit(PWP, self.logit_eps_)
                    X = LI.transpose(0, 1).cpu()
                else:
                    SS = MP[:, :, fc] - MP[:, :, sc]
                    if SS.dim() == 3:
                        SS = SS.squeeze()
                    X = SS.transpose(0, 1)

                PP = self.ldas_[fc][sc].predict_proba(X.detach().cpu())
                p_probs[:, sc, fc] = torch.from_numpy(PP[:, 0])
                p_probs[:, fc, sc] = torch.from_numpy(PP[:, 1])

        end = timer()
        print("Predict proba finished in " + str(end - start) + " s")

        return PWComb(p_probs.to(device=self.dev_, dtype=self.dtp_), verbose=debug_pwcm)

    def test_pairwise(self, MP, tar):
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                # Obtains fc and sc probabilities for samples belonging to those classes
                SS = MP[:, (tar == fc) + (tar == sc)][:, :, [fc, sc]].to(device=self.dev_, dtype=self.dtp_)
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
        """
        Combines outputs of constituent classifiers using only those classes, which are among the top l most probable
        for some constituent classifier.
        :param MP: MP: c x n x k tensor of constituent classifiers posteriors
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param l: how many most probable classes for each constituent classifier to consider
        :param PWComb: coupling method to use
        :return: n x k tensor of combined posteriors
        """
        print("Starting predict proba topl")
        if self.trained_on_penultimate_ is None:
            print("Ensemble not trained")
            return
        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_

        if not self.trained_on_penultimate_:
            num_non_one = torch.sum(torch.abs(torch.sum(MP, dim=2) - 1.0) > self.logit_eps_).item()
            if num_non_one > 0:
                print("Warning: " + str(num_non_one) +
                      " samples with non unit sum of supports found, performing softmax")
                MP = self.softmax_supports(MP)

        ps = torch.zeros(n, k, device=self.dev_, dtype=self.dtp_)
        # Every sample may have different set of top classes, so we process them one by one
        print_step = n // 100
        for ni in range(n):
            if ni % print_step == 0:
                print("Predicting proba topl progress " + str(ni//print_step) + "%", end="\r")

            val, ind = torch.topk(MP[:, ni, :], l, dim=1)
            Ti = sorted(list(set(ind.flatten().tolist())))
            tcc = len(Ti)
            p_probs = torch.zeros(1, tcc, tcc, device=self.dev_, dtype=self.dtp_)
            for fci, fc in enumerate(Ti):
                for sci, sc in enumerate(Ti[fci + 1:]):
                    if not self.trained_on_penultimate_:
                        # Obtains fc and sc probabilities for current sample
                        SS = MP[:, ni, [fc, sc]].to(device=self.dev_, dtype=self.dtp_)
                        # Computes p_ij pairwise probabilities for above mentioned samples
                        PWP = torch.true_divide(SS[:, 0], torch.sum(SS, 1) + (SS[:, 0] == 0))
                        LI = logit(PWP, self.logit_eps_)
                        X = LI.T.unsqueeze(0).cpu()
                    else:
                        SS = MP[:, ni, fc] - MP[:, ni, sc]
                        X = SS.T.unsqueeze(0)
                    PP = self.ldas_[fc][sc].predict_proba(X)
                    p_probs[:, fci + 1 + sci, fci] = torch.from_numpy(PP[:, 0])
                    p_probs[:, fci, fci + 1 + sci] = torch.from_numpy(PP[:, 1])

            sam_ps = PWComb(p_probs.to(device=self.dev_, dtype=self.dtp_))
            ps[ni, Ti] = sam_ps.squeeze()

        end = timer()
        print("Predict proba topl finished in " + str(end - start) + " s")

        return ps

    def predict_proba_topl_fast(self, MP, l, PWComb):
        """
        Better optimized version of predict_proba_topl
        Combines outputs of constituent classifiers using only those classes, which are among the top l most probable
        for some constituent classifier.
        :param MP: MP: c x n x k tensor of constituent classifiers posteriors
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param l: how many most probable classes for each constituent classifier to consider
        :param PWComb: coupling method to use
        :return: n x k tensor of combined posteriors
        """
        print("Starting predict proba topl fast")
        if self.trained_on_penultimate_ is None:
            print("Ensemble not trained")
            return
        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_

        if not self.trained_on_penultimate_:
            num_non_one = torch.sum(torch.abs(torch.sum(MP, dim=2) - 1.0) > self.logit_eps_).item()
            if num_non_one > 0:
                print("Warning: " + str(num_non_one) +
                      " samples with non unit sum of supports found, performing softmax")
                MP = self.softmax_supports(MP)

        MP = MP.to(device=self.dev_, dtype=self.dtp_)
        # ind is c x n x l tensor of top l indices for each sample in each network output
        val, ind = torch.topk(MP, l, dim=2)
        M = torch.zeros(MP.shape, dtype=torch.bool, device=self.dev_)
        # place true in positions of top probs
        # c x n x k tensor
        M.scatter_(2, ind, True)
        # combine selections over c inputs
        # n x k tensor containing for each sample a mask of union of top l classes from each constituent classifier
        M = torch.sum(M, dim=0, dtype=torch.bool)
        # zeroe lower values
        # c x n x k tensor
        MPz = MP * M
        # n x c x k tensor
        MPz.transpose_(0, 1)
        ps = torch.zeros(n, k, device=self.dev_, dtype=self.dtp_)
        # Selected class counts for every n
        NPC = torch.sum(M, 1).squeeze()
        # goes over possible numbers of classes in union of top l classes from each constituent classifier
        for pc in range(l, l * c + 1):
            # Pick those samples which have pc classes in the union
            # pcn x c x k tensor
            pcMPz = MPz[NPC == pc]
            # Pick pc-class masks
            # pcn x k tensor
            pcM = M[NPC == pc]
            # Number of samples with pc classes in the union
            pcn = pcM.shape[0]
            if pcn == 0:
                continue

            # One dimensional tensors of row and column indices of elements in upper triangle of pc x pc matrix,
            # ordered from left to right from top to bottom
            pcIMR = torch.tensor([], dtype=torch.long, device=self.dev_)
            pcIMC = torch.tensor([], dtype=torch.long, device=self.dev_)
            for r in range(pc):
                pcIMR = torch.cat((pcIMR, torch.tensor([r], dtype=torch.long, device=self.dev_).repeat(pc - r - 1)))
                pcIMC = torch.cat((pcIMC, torch.arange(r + 1, pc, device=self.dev_)))

            # pcn x pc tensor containing for each of the pcn samples indices of classes belonging to the union
            pcMi = torch.nonzero(pcM, as_tuple=False)[:, 1].view(pcM.shape[0], pc)
            # Only the values for classes in the union
            # pcn x c x pc tensor
            pcMPp = pcMPz.gather(2, pcMi.unsqueeze(1).expand(pcn, c, pc))

            # For every pcn and c contains values of top right triangle of matrix,
            # formed from supports expanded over columns, from left to right, from top to bottom
            # pcn x c x pc * (pc - 1) // 2 tensor
            pcMPpR = pcMPp[:, :, pcIMR]
            # For every pcn and c contains values of top right triangle of transposed matrix,
            # formed from supports expanded over columns, from left to right, from top to bottom
            # pcn x c x pc * (pc - 1) // 2 tensor
            pcMPpC = pcMPp[:, :, pcIMC]
            if not self.trained_on_penultimate_:
                # For every pcn and c contains top right triangle of pairwise probs p_ij
                # pcn x c x pc * (pc - 1) // 2 tensor
                pcPWP = pcMPpR / (pcMPpR + pcMPpC)

                # logit pairwise probs
                # pcn x c x pc * (pc - 1) // 2 tensor
                pcLI = logit(pcPWP, self.logit_eps_)
            else:
                # compute instead as a difference of supports
                # pcn x c x pc * (pc - 1) // 2 tensor
                pcLI = pcMPpR - pcMPpC

            # Flattened logits in order of dimensions: pcn; pc x pc top right triangle by rows; c
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
            dI0 = torch.arange(0, pcn, device=self.dev_, dtype=self.dtp_).repeat_interleave(val_ps)
            dI1 = pcIMR.repeat(pcn)
            dI2 = pcIMC.repeat(pcn)

            I = torch.cat((dI0.unsqueeze(0), dI1.unsqueeze(0), dI2.unsqueeze(0)), 0)
            DPS = torch.sparse_coo_tensor(I, CPWP, (pcn, pc, pc), device=self.dev_, dtype=self.dtp_).to_dense()
            It = torch.cat((dI0.unsqueeze(0), dI2.unsqueeze(0), dI1.unsqueeze(0)), 0)
            DPSt = torch.sparse_coo_tensor(It, 1 - CPWP, (pcn, pc, pc), device=self.dev_, dtype=self.dtp_).to_dense()
            # DPS now should contain pairwise probabilities
            DPS = DPS + DPSt

            pcPS = PWComb(DPS)

            # resulting posteriors for samples with pc picked classes
            ps_cur = torch.zeros(pcn, k, device=self.dev_, dtype=self.dtp_)
            row_mask = (NPC == pc)
            ps_cur[M[row_mask]] = torch.flatten(pcPS)
            # Insert current results into complete tensor of posteriors
            ps[row_mask] = ps_cur

        end = timer()
        print("Predict proba topl fast finished in " + str(end - start) + " s")

        return ps

    def save(self, file):
        """
        Save trained lda models into a file.
        :param file: file to save the models to
        :return:
        """
        print("Saving models into file: " + str(file))
        dump_dict = {"ldas": self.ldas_, "on_penult": self.trained_on_penultimate_}
        with open(file, 'wb') as f:
            pickle.dump(dump_dict, f)

    def load(self, file):
        """
        Load trained lda models from a file
        :param file: file to load the models from
        :return:
        """
        print("Loading models from file: " + str(file))
        with open(file, 'rb') as f:
            dump_dict = pickle.load(f)
            self.ldas_ = dump_dict["ldas"]
            self.trained_on_penultimate_ = dump_dict["on_penult"]

        self.k_ = len(self.ldas_)
        if self.k_ > 0:
            self.c_ = len(self.ldas_[0][1].coef_[0])

        self.coefs_ = torch.zeros(self.k_, self.k_, self.c_ + 1, device=self.dev_, dtype=self.dtp_)

        for fc in range(len(self.ldas_)):
            for sc in range(fc + 1, len(self.ldas_[fc])):
                clf = self.ldas_[fc][sc]
                self.coefs_[fc, sc, :] = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                                                    torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))

    def save_coefs_csv(self, file):
        """
        Save trained lda coefficients into a csv file.
        :param file: file to save the coefficients to
        :return:
        """
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
        """
        Save normality test p-values into a file, if these were computed during the training.
        :param file: file to save the p-values to
        :return:
        """
        print("Saving pvals into file: " + str(file))
        if self.pvals_ is not None:
            np.save(file, self.pvals_)
        else:
            print("P-values not computed")

    def set_averaging_weights(self):
        """
        Set the lda weights equal to one.
        :return:
        """
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                self.coefs_[fc, sc, :] = torch.tensor([1]*self.c_ + [0])

    def softmax_supports(self, MP):
        """
        Performs softmax on posteriors
        :param MP: c x n x k tensor of class supports
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :return:c x n x k tensor of posteriors
        """
        if self.dev_.type == 'cpu':
            return torch.nn.Softmax(dim=2)(MP)

        return torch.nn.Softmax(dim=2)(MP.to(device=self.dev_, dtype=self.dtp_)).cpu()
