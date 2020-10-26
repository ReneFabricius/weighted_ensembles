from my_codes.weighted_ensembles.SimplePWCombine import m1, m2, bc
from my_codes.weighted_ensembles.general_test import test_folder

test_folder("D:\\skola\\1\\weighted_ensembles\\my_codes\\tests\\test_data", "y_val.npy",
            "D:\\skola\\1\\weighted_ensembles\\my_codes\\tests\\outputs_fast_t1", [m1, m2, bc],
            models_load_file="D:\\skola\\1\\weighted_ensembles\\my_codes\\tests\\outputs\\models",
            combining_topl=5, testing_topk=1)
