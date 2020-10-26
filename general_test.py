import os
import torch
import numpy as np
from my_codes.weighted_ensembles.test_cifar10 import compute_acc_topk
from my_codes.weighted_ensembles.WeightedEnsemble import WeightedEnsemble


def test_folder(folder, targets, output_folder, comb_methods, models_load_file=None, combining_topl=5, testing_topk=1):
    extension = '.npy'
    models_file = 'models'
    npy_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[-1] == extension]
    if targets not in npy_files:
        raise NameError("Targets file not found")

    npy_files.remove(targets)

    def process_file(n):
        M = torch.tensor(np.load(os.path.join(folder, n)), dtype=torch.float32)
        return M.unsqueeze(0)

    tcs = torch.cat(list(map(process_file, np.array(npy_files))), 0)

    tar = torch.tensor(np.load(os.path.join(folder, targets)))

    c, n, k = tcs.size()

    print("Number of inputs detected: " + str(c))

    for nni in range(c):
        acci = compute_acc_topk(tar.cuda(), tcs[nni].cuda(), testing_topk)
        print("Accuracy of input (topk " + str(testing_topk) + ") " + str(npy_files[nni]) + ": " + str(acci))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    WE = WeightedEnsemble(c, k, device)
    if models_load_file is None:
        WE.fit(tcs, tar, False)
        WE.save_models(os.path.join(output_folder, models_file))
    else:
        WE.load_models(models_load_file)

    print("Working with topl: " + str(combining_topl) + " and topk: " + str(testing_topk))

    with torch.no_grad():
        for cm in comb_methods:
            print("Testing combining method " + cm.__name__)
            if combining_topl > 0:
                PPtl = WE.predict_proba_topl_fast(tcs, combining_topl, cm)
            else:
                PPtl = WE.predict_proba(tcs, cm)

            np.save(os.path.join(output_folder, "prob_" + cm.__name__), PPtl.cpu())
            acc = compute_acc_topk(tar.cuda(), PPtl, testing_topk)
            print("Accuracy of model: " + str(acc))

    return 0
