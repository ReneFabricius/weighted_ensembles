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
    def __init__(self, c, k):
        self.logit_eps_ = 0.00001
        self.c_ = c
        self.k_ = k
        self.coefs_ = [[[] for j in range(i + 1, k)] for i in range(k - 1)]
        self.ldas_ = [[None for j in range(i + 1, k)] for i in range(k - 1)]

    def fit(self, MP, tar):
        class_indexes = []
        for ki in range(self.k_):
            class_indexes.append(torch.nonzero((tar == ki), as_tuple=False).unsqueeze(0))

        # Indexes of samples belonging to each class, each row corresponds to one class
        CI = torch.cat(class_indexes, 0).cuda()

        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                # Obtains fc and sc probabilities for samples belonging to those clases
                SS = MP[:, (tar == fc) + (tar == sc), [fc, sc]].cuda()
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



















def compute_lda_coefficients(tcs):
    c, n, k = tcs.size()
    TP = torch.nn.Softmax(dim=2)(tcs)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    y_test = torch.tensor(testset.targets)

    per_class = n / k
    class_indexes = []
    for ki in range(k):
        class_indexes.append(torch.nonzero((y_test == ki), as_tuple=False).unsqueeze(0))

    # Indexes of samples belonging to each class, each row corresponds to one class
    CI = torch.cat(class_indexes, 0).cuda()

    val_portion = 0.5
    val_number = int(per_class*val_portion)

    TP_val = TP[:, torch.cat([CI[ci, :val_number] for ci in range(k)], 0), :]
    #TP_test = TP[:, torch.cat([CI[ci, val_number:] for ci in range(k)], 0), :]
    tar_val = [torch.cat([torch.Tensor(val_number).fill_(ci) for ci in range(k)])]
    #tar_test = [torch.cat([torch.Tensor(per_class - val_number).fill_(ci) for ci in range(k)])]

    lda_coefs = []
    p_probs = torch.Tensor(n, k, k)

    for fc in range(k):
        for sc in range(fc + 1, k):
            # Obtains probs for classes fc and sc of the first val_number samples belonging to each of those classes
            SS_val = TP_val[:, (tar_val == fc) + (tar_val == sc), [fc, sc]].cuda()
            # Computes p_ij pairwise probabilities for above mentioned samples
            PWP_val = torch.true_divide(SS_val[:, :, 0], torch.sum(SS_val, 2))
            LI_val = logit(PWP_val, 0.00001)
            X_val = LI_val.transpose(0, 1).cpu()
            y = tar_val[(tar_val == fc) + (tar_val == sc)]
            y[y == fc] = 1
            y[y == sc] = 0
            clf = LinearDiscriminantAnalysis()

            clf.fit(X_val, y)

            lda_coefs.append(np.concatenate([[fc, sc], np.squeeze(clf.coef_), clf.intercept_], axis=0))


            PP = clf.predict_proba(X_val)
            p_probs[:, fc, sc] = torch.from_numpy(PP[:, 0])
            p_probs[:, sc, fc] = torch.from_numpy(PP[:, 1])

    np.save('coeffs', np.array(lda_coefs))
    np.save('probs', np.array(p_probs))
    return p_probs, clf


fold = "D:\\skola\\1\\weighted_ensembles\\my_codes\\weighted_ensembles\\predictions"


def process_file(n):
    M = torch.tensor(np.load(os.path.join(fold, n)), dtype=torch.float32)
    return M.unsqueeze(0)


def main():
    names = ["CNN_dropout_159.npy", "fast_cifar10_prob-e9.npy", "SimpleNet_test_output.npy"]
    inp = torch.cat(list(map(process_file,  names)), 0)
    return compute_lda_coefficients(inp)
