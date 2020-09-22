import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms
from my_codes.weighted_ensembles.WeightedEnsemble import WeightedEnsemble
from my_codes.weighted_ensembles.SimplePWCombine import m1, m2


def compute_acc_topk(y_cor, ps, l):
    top_v, top_i = torch.topk(ps, l, dim=1)
    n = y_cor.size()[0]

    return torch.sum(top_i == y_cor.unsqueeze(1)).item() / n


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

    WE = WeightedEnsemble(c, k, m1)
    WE.fit(TP_val, tar_val)
    PP = WE.predict_proba(TP_test)

    acc = compute_acc_topk(tar_test, PP, 1)

    return acc

