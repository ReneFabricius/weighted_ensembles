import os
import torch
import numpy as np
from my_codes.weighted_ensembles.test_cifar10 import compute_acc_topk
from my_codes.weighted_ensembles.WeightedEnsemble import WeightedEnsemble
from my_codes.weighted_ensembles.SimplePWCombine import m1, m2, bc


def test_imagenet():
    fold = "D:\\skola\\5\\prax\\image_net\\probs"
    names = ['DenseNet121.npy', 'DenseNet169.npy', 'DenseNet201.npy', 'MobileNet.npy', 'MobileNetV2.npy',
             'NASNetMobile.npy', 'ResNet101.npy', 'ResNet101V2.npy', 'ResNet152.npy', 'ResNet152V2.npy',
             'ResNet50.npy', 'ResNet50V2.npy', 'VGG16.npy', 'VGG19.npy', 'Xception.npy']

    labels = 'y_val.npy'
    models_file = "D:\\skola\\1\\weighted_ensembles\\models_file"

    inds_to_combine = [0, 9, 13]

    def process_file(n):
        M = torch.tensor(np.load(os.path.join(fold, n)), dtype=torch.float32)
        return M.unsqueeze(0)

    tcs = torch.cat(list(map(process_file, np.array(names)[inds_to_combine])), 0)

    tar = torch.tensor(np.load(os.path.join(fold, labels)))

    c, n, k = tcs.size()

    for nni in range(c):
        acci = compute_acc_topk(tar.cuda(), tcs[nni].cuda(), 1)
        print("Accuracy of network " + str(names[inds_to_combine[nni]]) + ": " + str(acci))

    WE = WeightedEnsemble(c, k)
    #WE.fit(tcs, tar, False)

    #WE.save_models(models_file)

    WE.load_models(models_file)


    with torch.no_grad():
        PPtl = WE.predict_proba_topl(tcs, 5, bc)


    acctl = compute_acc_topk(tar.cuda(), PPtl, 1)

    print("Accuracy of topl model: " + str(acctl))

    return acctl, PPtl


test_imagenet()