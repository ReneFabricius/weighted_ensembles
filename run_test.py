from my_codes.weighted_ensembles.SimplePWCombine import m1, m2, bc
from my_codes.weighted_ensembles.general_test import test_folder

test_folder(
    train_folder="D:\\skola\\1\\weighted_ensembles\\tests_imagenet\\test_data",
    test_folder="D:\\skola\\1\\weighted_ensembles\\tests_imagenet\\test_data",
    targets="y_val.npy",
    order="order.txt",
    output_folder="D:\\skola\\1\\weighted_ensembles\\tests_imagenet\\outputs_fast_t1",
    output_model_fold="D:\\skola\\1\\weighted_ensembles\\tests_imagenet\\outputs_fast_t1",
    comb_methods=[m1, m2, bc],
    models_load_file="D:\\skola\\1\\weighted_ensembles\\tests_imagenet\\outputs\\models",
    combining_topl=5,
    testing_topk=1)
