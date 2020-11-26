from my_codes.weighted_ensembles.SimplePWCombine import m1, m2, bc
from my_codes.weighted_ensembles.general_test import test_folder

test_folder(
    train_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_val_proba",
    test_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_val_proba",
    targets="y_val.npy",
    order="order.txt",
    output_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\out_test_val",
    output_model_fold="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model",
    comb_methods=[m1, m2, bc],
    models_load_file=None,
    combining_topl=2,
    testing_topk=1,
    save_coefs=True)

test_folder(
    train_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_val_proba",
    test_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\test-6",
    targets="y_val.npy",
    order="order.txt",
    output_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\out_test-6",
    output_model_fold="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model",
    comb_methods=[m1, m2, bc],
    models_load_file="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model\\models",
    combining_topl=2,
    testing_topk=1)

test_folder(
    train_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_val_proba",
    test_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\test-3",
    targets="y_val.npy",
    order="order.txt",
    output_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\out_test-3",
    output_model_fold="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model",
    comb_methods=[m1, m2, bc],
    models_load_file="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model\\models",
    combining_topl=2,
    testing_topk=1)

test_folder(
    train_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_val_proba",
    test_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\test0",
    targets="y_val.npy",
    order="order.txt",
    output_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\out_test0",
    output_model_fold="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model",
    comb_methods=[m1, m2, bc],
    models_load_file="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model\\models",
    combining_topl=2,
    testing_topk=1)

test_folder(
    train_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_val_proba",
    test_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\test3",
    targets="y_val.npy",
    order="order.txt",
    output_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\out_test3",
    output_model_fold="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model",
    comb_methods=[m1, m2, bc],
    models_load_file="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model\\models",
    combining_topl=2,
    testing_topk=1)

test_folder(
    train_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_val_proba",
    test_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\test6",
    targets="y_val.npy",
    order="order.txt",
    output_folder="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\eval_mavd_3_proba\\out_test6",
    output_model_fold="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model",
    comb_methods=[m1, m2, bc],
    models_load_file="D:\\skola\\1\\weighted_ensembles\\tests_spectro\\model\\models",
    combining_topl=2,
    testing_topk=1)