import os
import torch
import numpy as np
import functools
from weensembles.predictions_evaluation import compute_acc_topk
from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble


def ensemble_general_test(data_train_path, data_test_path, targets, order, output_folder, output_model_fold, comb_methods,
                          models_load_file=None, combining_topl=5, testing_topk=1, save_coefs=False, verbose=False,
                          test_normality=False, save_pvals=False, fit_on_penultimate=False, double_precision=False):
    """
    Trains a combiner on provided networks outputs - data_train_path (or loads models if models_load_file is provided)
    and tests it on outputs in data_test_path

    :param save_pvals:
    :param test_normality:
    :param data_train_path: folder containing outputs of networks for training the combiner,
    respective targets and order file
    :param data_test_path: folder containing outputs of networks for testing the combiner,
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
    dtype = (torch.float64 if double_precision else torch.float32)
    print("Using device: " + str(device))
    print("Using dtype: " + str(dtype))

    def process_file(n, fold):
        M = torch.tensor(np.load(os.path.join(fold, n)), dtype=dtype)
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
        print("Training from folder: " + data_train_path)
        npy_files_train = check_inputs(data_train_path)

        # Load networks' training outputs
        tcs = torch.cat(list(map(functools.partial(process_file, fold=data_train_path), np.array(npy_files_train))), 0)
        # Load training targets
        tar = torch.tensor(np.load(os.path.join(data_train_path, targets)))
        # c-number of networks, n-number of classified instances, k-number of classes
        c, n, k = tcs.size()

        print("Number of inputs detected: " + str(c))

        # Compute accuracies of individual networks on training data
        for nni in range(c):
            acci = compute_acc_topk(tar.to(device=device, dtype=dtype), tcs[nni].to(device=device, dtype=dtype), testing_topk)
            print("Accuracy of train input (topk " + str(testing_topk) + ") " + str(npy_files_train[nni]) +
                  ": " + str(acci))

        WE = WeightedLinearEnsemble(c=c, k=k, device=device, dtp=dtype)

        if not fit_on_penultimate:
            WE.fit(tcs, tar, verbose, test_normality)
        else:
            WE.fit_penultimate(tcs, tar, verbose, test_normality)
        WE.save(os.path.join(output_model_fold, models_file))
        if save_pvals:
            WE.save_pvals(os.path.join(output_folder, "p_values.npy"))
    else:
        WE = WeightedLinearEnsemble(device=device, dtp=dtype)
        WE.load(models_load_file)
        c = WE.c_
        k = WE.k_

    if save_coefs:
        WE.save_coefs_csv(os.path.join(output_model_fold, "lda_coefs.csv"))

    print("Working with topl: " + str(combining_topl) + " and topk: " + str(testing_topk))

    # Testing
    # Check existence of files
    print("Testing from folder: " + data_test_path)
    npy_files_test = check_inputs(data_test_path, check_tar=False)

    # Load networks' testing outputs
    tcs_test = torch.cat(list(map(functools.partial(process_file, fold=data_test_path), np.array(npy_files_test))), 0)

    has_test_tar = os.path.isfile(os.path.join(data_test_path, targets))
    if has_test_tar:
        tar_test = torch.tensor(np.load(os.path.join(data_test_path, targets)))

        # Compute accuracies of individual networks on testing data
        for nni in range(c):
            acci = compute_acc_topk(tar_test.to(device=device, dtype=dtype),
                                    tcs_test[nni].to(device=device, dtype=dtype), testing_topk)
            print("Accuracy of test input (topk " + str(testing_topk) + ") " + str(npy_files_test[nni]) + ": " + str(acci))

    with torch.no_grad():
        for cm in comb_methods:
            print("Testing combining method " + cm.__name__)
            if combining_topl > 0:
                fin = False
                tries = 0
                cur_n = tcs_test.shape[1]
                while not fin and tries < 20 and cur_n > 0:
                    if tries > 0:
                        torch.cuda.empty_cache()
                        print('Trying again, try {}, batch size {}'.format(tries, cur_n))
                    try:
                        PPtl = WE.predict_proba_topl_fast(tcs_test, combining_topl, cm, batch_size=cur_n)
                        fin = True
                    except RuntimeError as rerr:
                        if 'memory' not in str(rerr):
                            raise rerr
                        print("OOM Exception")
                        del rerr
                        cur_n = cur_n // 2
                        tries += 1

                if not fin:
                    print('Unsuccessful')
                    return -1

            else:
                PPtl = WE.predict_proba(tcs_test, cm)

            np.save(os.path.join(output_folder, "prob_" + cm.__name__), PPtl.cpu())
            if has_test_tar:
                acc = compute_acc_topk(tar_test.to(device=device, dtype=dtype), PPtl, testing_topk)
                print("Accuracy of model: " + str(acc))

    return 0


def test_averaging_combination(data_test_path, targets, order, output_folder, comb_methods,
                               combining_topl=5, testing_topk=1):
    """
    Tests simple averaging ensemble on provided test outputs in folder data_test_path.

    :param data_test_path: folder containing outputs of networks for testing the combiner,
    alternatively also testing targets
    :param targets: name of targets file
    :param order: text folder containing names of networks' output files to be combined
    :param output_folder: folder into which ensemble outputs will be saved
    :param comb_methods: list of combining methods to test
    :param combining_topl: number of top classes for each network which will be combined,
    enter zero to combine all classes
    :param testing_topk: number of top classes which will be considered in accuracy computation
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print("Using device: " + str(device))
    print("Using dtype: " + str(dtype))

    def process_file(n, fold):
        M = torch.tensor(np.load(os.path.join(fold, n)), dtype=dtype)
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

    print("Working with topl: " + str(combining_topl) + " and topk: " + str(testing_topk))

    # Testing
    # Check existence of files
    print("Testing from folder: " + data_test_path)
    npy_files_test = check_inputs(data_test_path, check_tar=False)

    # Load networks' testing outputs
    tcs_test = torch.cat(list(map(functools.partial(process_file, fold=data_test_path), np.array(npy_files_test))), 0)

    c, n, k = tcs_test.size()
    WE = WeightedLinearEnsemble(c=c, k=k, device=device, dtp=dtype)
    WE.set_averaging_weights()

    has_test_tar = os.path.isfile(os.path.join(data_test_path, targets))
    if has_test_tar:
        tar_test = torch.tensor(np.load(os.path.join(data_test_path, targets)))

        # Compute accuracies of individual networks on testing data
        for nni in range(c):
            acci = compute_acc_topk(tar_test.to(device=device, dtype=dtype),
                                    tcs_test[nni].to(device=device, dtype=dtype), testing_topk)
            print("Accuracy of test input (topk " + str(testing_topk) + ") " + str(npy_files_test[nni]) + ": " + str(
                acci))

    with torch.no_grad():
        for cm in comb_methods:
            print("Testing combining method " + cm.__name__)
            if combining_topl > 0:
                PPtl = WE.predict_proba_topl_fast(tcs_test, combining_topl, cm)
            else:
                PPtl = WE.predict_proba(tcs_test, cm)

            np.save(os.path.join(output_folder, "prob_" + cm.__name__), PPtl.cpu())
            if has_test_tar:
                acc = compute_acc_topk(tar_test.to(device=device, dtype=dtype), PPtl, testing_topk)
                print("Accuracy of model: " + str(acc))

    return 0

