import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms


def compute_lda_coefficients(tcs):
    c, n, k = tcs.size()
    tcs_prob = torch.nn.Softmax(dim=2)(tcs)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    y_test = torch.tensor(testset.targets)

    per_class = n / k
    class_indexes = []
    for ki in range(k):
        class_indexes.append((y_test == ki).nonzero())

    lda_coefs = []

    for fc in range(k):
        for sc in range(fc + 1, k):
            






    '''
    batch_size = n

    E = torch.eye(k).cuda()
    VE = (1 - E).unsqueeze(0).unsqueeze(1).expand(c, batch_size, k, k)

    bn = 0
    for si in range(0, n, batch_size):
        print('Batch: ' + str(bn))
        # print(torch.cuda.memory_summary(0))
        tcsp = tcs[:, si: si + batch_size, :].cuda()

        # last batch may have smaller size
        if tcsp.size()[1] < batch_size:
            VE = VE[:, 0:tcsp.size()[1], :, :]

        # four dimensional tensor, first d - input classifier index,
        # second d - index of sample in batch, last two d - matrices
        # with columns filled by respective prob vector and zero diagonal
        TCs = VE*tcsp.unsqueeze(3)

        # four dimensional tensor, first d - input classifier index,
        # second d - index of sample in batch, last two d - matrices
        # with pairwise probabilities
        PWP = TCs / (TCs + TCs.transpose(2, 3) + (TCs == 0))

        bn += 1
        '''



fold = "D:\\skola\\1\\weighted_ensembles\\my_codes\\weighted_ensembles\\predictions"


def process_file(n):
    M = torch.tensor(np.load(os.path.join(fold, n)), dtype=torch.float32)
    return M.unsqueeze(0)


def main():
    names = ["CNN_dropout_159.npy", "fast_cifar10_prob-e9.npy", "SimpleNet_test_output.npy"]
    inp = torch.cat(list(map(process_file,  names)), 0)
    return compute_lda_coefficients(inp)

