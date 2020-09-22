import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def logit(T, eps):
    EPS = T.new_full(T.shape, eps)
    L = torch.where(T < eps, EPS, T)
    LU = torch.where(L > 1 - eps, 1 - EPS, L)
    return torch.log(LU/(1 - LU))


class WeightedEnsemble:
    def __init__(self, c, k, PWComb):
        self.logit_eps_ = 0.00001
        self.c_ = c
        self.k_ = k
        self.coefs_ = [[[] for j in range(k)] for i in range(k)]
        self.ldas_ = [[None for j in range(k)] for i in range(k)]
        self.PWC_ = PWComb

    def fit(self, MP, tar):
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                # Obtains fc and sc probabilities for samples belonging to those classes
                SS = MP[:, (tar == fc) + (tar == sc)][:, :, [fc, sc]].cuda()
                # Computes p_ij pairwise probabilities for above mentioned samples
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2))
                LI = logit(PWP, self.logit_eps_)
                X = LI.transpose(0, 1).cpu()
                y = tar[(tar == fc) + (tar == sc)]
                y[y == fc] = 1
                y[y == sc] = 0
                clf = LinearDiscriminantAnalysis()

                clf.fit(X, y)

                self.ldas_[fc][sc] = clf
                self.coefs_[fc][sc] = [clf.coef_, clf.intercept_]

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
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2))
                LI = logit(PWP, self.logit_eps_)
                X = LI.transpose(0, 1).cpu()
                PP = self.ldas_[fc][sc].predict_proba(X)
                p_probs[:, fc, sc] = torch.from_numpy(PP[:, 0])
                p_probs[:, sc, fc] = torch.from_numpy(PP[:, 1])

        return self.PWC_(p_probs)
