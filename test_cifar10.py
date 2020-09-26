import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms
from my_codes.weighted_ensembles.WeightedEnsemble import WeightedEnsemble
from my_codes.weighted_ensembles.SimplePWCombine import m1, m2, bc


def compute_acc_topk(y_cor, ps, l):
    top_v, top_i = torch.topk(ps, l, dim=1)
    n = y_cor.size()[0]

    return torch.sum(top_i == y_cor.unsqueeze(1)).item() / n


def create_pairwise(P):
    n, k = P.size()
    E = torch.eye(k).cuda()
    VE = (1 - E).unsqueeze(0).expand(n, k, k)
    TCs = VE * P.cuda().unsqueeze(2)
    R = TCs / (TCs + TCs.transpose(1, 2) + (TCs == 0))

    return R


def test_cifar10():
    fold = "D:\\skola\\1\\weighted_ensembles\\my_codes\\weighted_ensembles\\predictions"

    def process_file(n):
        M = torch.tensor(np.load(os.path.join(fold, n)), dtype=torch.float32)
        return M.unsqueeze(0)

    names = ["CNN_dropout_159.npy", "fast_cifar10_prob-e9.npy", "SimpleNet_test_output.npy"]
    tcs = torch.cat(list(map(process_file, names)), 0)

    c, n, k = tcs.size()
    TP = torch.nn.Softmax(dim=2)(tcs)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    y_test = torch.tensor(testset.targets)

    per_class = n // k
    class_indexes = []
    for ki in range(k):
        class_indexes.append(torch.nonzero((y_test == ki), as_tuple=False).transpose(0, 1))

    # Indexes of samples belonging to each class, each row corresponds to one class
    CI = torch.cat(class_indexes, 0).cuda()

    val_portion = 0.5
    val_number = int(per_class * val_portion)

    TP_val = TP[:, torch.cat([CI[ci, :val_number] for ci in range(k)], 0), :]
    TP_test = TP[:, torch.cat([CI[ci, val_number:] for ci in range(k)], 0), :]
    tar_val = torch.cat([torch.Tensor(val_number).fill_(ci) for ci in range(k)])
    tar_test = torch.cat([torch.Tensor(per_class - val_number).fill_(ci) for ci in range(k)])

    for nni in range(c):
        acci = compute_acc_topk(tar_test.cuda(), TP_test[nni].cuda(), 1)
        print("Accuracy of network " + str(nni) + ": " + str(acci))
        '''
        pwtpi = create_pairwise(TP_test[nni].cuda())
        tp_rev = m1(pwtpi)
        accreci = compute_acc_topk(tar_test.cuda(), tp_rev.cuda(), 1)
        print("Accuracy of recombined network " + str(nni) + ": " + str(accreci))
        '''

    WE = WeightedEnsemble(c, k, bc)
    WE.fit(TP_val, tar_val, True)

    with torch.no_grad():
        PP, p_probs = WE.predict_proba(TP_test)

        PPtl = WE.predict_proba_topl(TP_test, 5)

    # WE.test_pairwise(TP_test, tar_test)

    acc = compute_acc_topk(tar_test.cuda(), PP, 1)
    acctl = compute_acc_topk(tar_test.cuda(), PPtl, 1)

    print("Accuracy of model: " + str(acc))
    print("Accuracy of topl model: " + str(acctl))

    return acc, PP

