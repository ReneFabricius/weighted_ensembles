import os
from itertools import combinations
import sys
sys.path.append("D:/skola/1/weighted_ensembles")

from my_codes.weighted_ensembles.general_test import test_folder
from my_codes.weighted_ensembles.SimplePWCombine import m1, m2, bc

folder = "D:/skola/1/weighted_ensembles/tests_IM2012"

train = "comb_train"
test = "test"
outputs_all = "combin_outputs_penultimate"
output = "output"
model = "model"
targets = "targets.npy"
order = "order.txt"
min_ens_size = 3
topls = [5, 10, 15, 20]

net_outputs = []
for file in os.listdir(os.path.join(folder, train)):
    if file.endswith(".npy") and file != targets:
        net_outputs.append(file)
        print("Network output found: " + file)

train_fold = os.path.join(folder, train)
test_fold = os.path.join(folder, test)
order_file = os.path.join(folder, train, order)
order_file_test = os.path.join(folder, test, order)

num_nets = len(net_outputs)
for sss in range(min_ens_size, num_nets + 1):
    print("Testing " + str(sss) + " network ensembles")
    for sub_set in combinations(net_outputs, sss):
        for topl in topls:
            print("Testing topl: " + str(topl))
            sub_set_name = '_'.join([s[0:4] for s in sub_set]) + "_topl_" + str(topl)
            outputs_fold = os.path.join(folder, outputs_all, output + "_" + sub_set_name)
            models_fold = os.path.join(folder, outputs_all, model + "_" + sub_set_name)
            if not os.path.exists(outputs_fold):
                os.makedirs(outputs_fold)
            if not os.path.exists(models_fold):
                os.makedirs(models_fold)

            order_fl = open(order_file, 'w')
            order_fl.write('\n'.join(sub_set))
            order_fl.close()

            order_fl_test = open(order_file_test, 'w')
            order_fl_test.write('\n'.join(sub_set))
            order_fl_test.close()

            try:
                test_folder(train_fold, test_fold, targets, order, outputs_fold, models_fold, [m1, m2, bc],
                            combining_topl=topl, save_coefs=True, verbose=False, test_normality=True, save_pvals=True,
                            fit_on_penultimate=True)

            finally:
                os.remove(order_file)
                os.remove(order_file_test)





