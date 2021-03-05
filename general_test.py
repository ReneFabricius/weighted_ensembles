import os
import torch
import numpy as np
import functools
from my_codes.weighted_ensembles.test_cifar10 import compute_acc_topk
from my_codes.weighted_ensembles.WeightedEnsemble import WeightedEnsemble


def test_folder(train_folder, test_folder, targets, order, output_folder, output_model_fold, comb_methods,
                models_load_file=None, combining_topl=5, testing_topk=1, save_coefs=False, verbose=False,
                test_normality=False):
    """
    Trains a combiner on provided networks outputs - train_folder (or loads models if models_load_file is provided)
    and tests it on outputs in test_folder

    :param train_folder: folder containing outputs of networks for training the combiner,
    respective targets and order file
    :param test_folder: folder containing outputs of networks for testing the combiner,
    alternatively also testing targets
    :param targets: name of targets file
    :param order: text folder containing names of networks' output files to be combined
    :param output_folder: folder into which ensemble outputs will be saved
    :param output_model_fold: folder into which model file will be saved
    :param comb_methods: list of combining methods to test
    :param models_load_file: if not None, a saved model file which will be used
    :param combining_topl: number of top classes for each network which will be combined,
    enter zero to combine all classes
    :param testing_topk: number of top classes which will be considered in accuracy computation
    :param save_coefs: save coefficients as text
    :param verbose:
    :return:
    """
    extension = '.npy'
    models_file = 'models'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    def process_file(n, fold):
        M = torch.tensor(np.load(os.path.join(fold, n)), dtype=torch.float32)
        return M.unsqueeze(0)

    def check_inputs(fold, check_tar=True):
        inp_files = []
        with open(os.path.join(fold, order)) as ord_f:
            for nn_file in ord_f:
                nn_file_strip = nn_file.strip()
                if not os.path.isfile(os.path.join(fold, nn_file_strip)):
                    raise NameError("Error: nn file: " + nn_file_strip + " not found")
                inp_files.append(nn_file_strip)

        if check_tar:
            if not os.path.isfile(os.path.join(fold, targets)):
                raise NameError("Targets file not found")

        return inp_files

    if models_load_file is None:
        print("Training from folder: " + train_folder)
        npy_files_train = check_inputs(train_folder)

        # Load networks' training outputs
        tcs = torch.cat(list(map(functools.partial(process_file, fold=train_folder), np.array(npy_files_train))), 0)
        # Load training targets
        tar = torch.tensor(np.load(os.path.join(train_folder, targets)))
        # c-number of networks, n-number of classified instances, k-number of classes
        c, n, k = tcs.size()

        print("Number of inputs detected: " + str(c))

        # Compute accuracies of individual networks on training data
        for nni in range(c):
            acci = compute_acc_topk(tar.cuda(), tcs[nni].cuda(), testing_topk)
            print("Accuracy of train input (topk " + str(testing_topk) + ") " + str(npy_files_train[nni]) +
                  ": " + str(acci))

        WE = WeightedEnsemble(c=c, k=k, device=device)

        WE.fit(tcs, tar, verbose, test_normality)
        WE.save_models(os.path.join(output_model_fold, models_file))
    else:
        WE = WeightedEnsemble(device=device)
        WE.load_models(models_load_file)
        c = WE.c_
        k = WE.k_

    if save_coefs:
        WE.save_coefs_csv(os.path.join(output_model_fold, "lda_coefs.csv"))

    print("Working with topl: " + str(combining_topl) + " and topk: " + str(testing_topk))

    # Testing
    # Check existence of files
    print("Testing from folder: " + test_folder)
    npy_files_test = check_inputs(test_folder, check_tar=False)

    # Load networks' testing outputs
    tcs_test = torch.cat(list(map(functools.partial(process_file, fold=test_folder), np.array(npy_files_test))), 0)

    has_test_tar = os.path.isfile(os.path.join(test_folder, targets))
    if has_test_tar:
        tar_test = torch.tensor(np.load(os.path.join(test_folder, targets)))

        # Compute accuracies of individual networks on testing data
        for nni in range(c):
            acci = compute_acc_topk(tar_test.cuda(), tcs_test[nni].cuda(), testing_topk)
            print("Accuracy of test input (topk " + str(testing_topk) + ") " + str(npy_files_test[nni]) + ": " + str(acci))

    with torch.no_grad():
        for cm in comb_methods:
            print("Testing combining method " + cm.__name__)
            if combining_topl > 0:
                PPtl = WE.predict_proba_topl_fast(tcs_test, combining_topl, cm)
            else:
                PPtl = WE.predict_proba(tcs_test, cm)

            np.save(os.path.join(output_folder, "prob_" + cm.__name__), PPtl.cpu())
            if has_test_tar:
                acc = compute_acc_topk(tar_test.cuda(), PPtl, testing_topk)
                print("Accuracy of model: " + str(acc))

    return 0
