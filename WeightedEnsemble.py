import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

from timeit import default_timer as timer


def logit(T, eps):
    EPS = T.new_full(T.shape, eps)
    L = torch.where(T < eps, EPS, T)
    LU = torch.where(L > 1 - eps, 1 - EPS, L)
    return torch.log(LU/(1 - LU))


def pairwise_precisions(SS, tar):
    c, n, k = SS.size()
    top_v, top_i = torch.topk(SS, 1, dim=2)
    ti = top_i.squeeze(dim=2)
    # Coding of target is switched. 1 for class on index 0 and 0 for class on index 1
    return torch.sum(ti.cpu() != tar, dim=1)/float(n)


class WeightedEnsemble:
    def __init__(self, c, k, PWComb):
        self.logit_eps_ = 1e-5
        self.c_ = c
        self.k_ = k
        self.coefs_ = [[[] for j in range(k)] for i in range(k)]
        self.ldas_ = [[None for j in range(k)] for i in range(k)]
        self.PWC_ = PWComb

    def fit(self, MP, tar, verbose=False):
        """Trains lda for every pair of classes"""
        print("Starting fit")
        num = self.k_*(self.k_ - 1)//2
        print_step = num // 100
        pi = 0
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                if pi % print_step == 0:
                    print("Fit progress " + str(pi // print_step) + "%", end="\r")
                # Obtains fc and sc probabilities for samples belonging to those classes
                SS = MP[:, (tar == fc) + (tar == sc)][:, :, [fc, sc]].cuda()
                # Computes p_ij pairwise probabilities for above mentioned samples
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                LI = logit(PWP, self.logit_eps_)
                X = LI.transpose(0, 1).cpu()
                y = tar[(tar == fc) + (tar == sc)]
                mask_fc = (y == fc)
                mask_sc = (y == sc)
                y[mask_fc] = 1
                y[mask_sc] = 0

                clf = LinearDiscriminantAnalysis(solver='lsqr')
                clf.fit(X, y)
                self.ldas_[fc][sc] = clf
                self.coefs_[fc][sc] = [clf.coef_, clf.intercept_]

                if verbose:
                    pwacc = pairwise_precisions(SS, y)
                    print("Training pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                          "\n\tpairwise accuracies: " + str(pwacc) +
                          "\n\tchosen coefficients: " + str(clf.coef_) +
                          "\n\tintercept: " + str(clf.intercept_))

                    print("\tcombined accuracy: " + str(clf.score(X, y)))

                pi += 1

    def predict_proba(self, MP):
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_
        p_probs = torch.Tensor(n, k, k)

        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                # Obtains fc and sc probabilities for all classes
                SS = MP[:, :, [fc, sc]].cuda()
                # Computes p_ij pairwise probabilities for above mentioned samples
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                LI = logit(PWP, self.logit_eps_)
                X = LI.transpose(0, 1).cpu()
                PP = self.ldas_[fc][sc].predict_proba(X)
                p_probs[:, sc, fc] = torch.from_numpy(PP[:, 0])
                p_probs[:, fc, sc] = torch.from_numpy(PP[:, 1])

        return self.PWC_(p_probs.cuda()), p_probs

    def test_pairwise(self, MP, tar):
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                # Obtains fc and sc probabilities for samples belonging to those classes
                SS = MP[:, (tar == fc) + (tar == sc)][:, :, [fc, sc]].cuda()
                # Computes p_ij pairwise probabilities for above mentioned samples
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                LI = logit(PWP, self.logit_eps_)
                X = LI.transpose(0, 1).cpu()
                y = tar[(tar == fc) + (tar == sc)]
                mask_fc = (y == fc)
                mask_sc = (y == sc)
                y[mask_fc] = 1
                y[mask_sc] = 0

                pwacc = pairwise_precisions(SS, y)
                print(
                    "Testing pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                    "\n\tpairwise accuracies: " + str(pwacc) +
                    "\n\tchosen coefficients: " + str(self.ldas_[fc][sc].coef_) +
                    "\n\tintercept: " + str(self.ldas_[fc][sc].intercept_))

                print("\tcombined accuracy: " + str(self.ldas_[fc][sc].score(X, y)))

    def predict_proba_topl(self, MP, l):
        print("Starting predict proba")
        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_

        ps = torch.zeros(n, k).cuda()
        # Every sample may have different set of top classes, so we process them one by one
        print_step = n // 100
        for ni in range(n):
            if ni % print_step == 0:
                print("Predicting proba progress " + str(ni//print_step) + "%", end="\r")

            val, ind = torch.topk(MP[:, ni, :], l, dim=1)
            Ti = sorted(list(set(ind.flatten().tolist())))
            tcc = len(Ti)
            p_probs = torch.zeros(1, tcc, tcc).cuda()
            for fci, fc in enumerate(Ti):
                for sci, sc in enumerate(Ti[fci + 1:]):
                    # Obtains fc and sc probabilities for current sample
                    SS = MP[:, ni, [fc, sc]].cuda()
                    # Computes p_ij pairwise probabilities for above mentioned samples
                    PWP = torch.true_divide(SS[:, 0], torch.sum(SS, 1) + (SS[:, 0] == 0))
                    LI = logit(PWP, self.logit_eps_)
                    X = LI.T.unsqueeze(0).cpu()
                    PP = self.ldas_[fc][sc].predict_proba(X)
                    p_probs[:, fci + 1 + sci, fci] = torch.from_numpy(PP[:, 0])
                    p_probs[:, fci, fci + 1 + sci] = torch.from_numpy(PP[:, 1])

            sam_ps = self.PWC_(p_probs.cuda())
            ps[ni, Ti] = sam_ps.squeeze()

        end = timer()
        print("Predict proba finished in " + str(end - start) + " s")

        return ps

    def save_models(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.ldas_, f)

    def load_models(self, file):
        with open(file, 'rb') as f:
            self.ldas_ = pickle.load(f)

