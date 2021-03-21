from my_codes.weighted_ensembles.predictions_evaluation import compute_acc_topk
from os import path, listdir
import pandas as pd
import numpy as np
import regex as re
import torch

nets_folder = "D:\\skola\\1\\weighted_ensembles\\tests_IM2012\\test"
combin_folder = "D:\\skola\\1\\weighted_ensembles\\tests_IM2012\\combin_outputs_penultimate"
targets = "targets.npy"
output_folder = "D:\\skola\\1\\weighted_ensembles\\tests_IM2012\\combin_evaluation_penultimate"


tar = torch.from_numpy(np.load(path.join(nets_folder, targets)))
computed_accuracies = [1, 5]
net_abbrevs = []
nets_df = pd.DataFrame(columns=('net', *['top' + str(k) for k in computed_accuracies]))
print("Processing nets folder {}".format(nets_folder))
for f in listdir(nets_folder):
    if path.splitext(f)[1] == '.npy' and f != targets:
        print("Found network {}".format(f))
        cur_net = torch.from_numpy(np.load(path.join(nets_folder, f)))
        accuracies = [compute_acc_topk(tar, cur_net, k) for k in computed_accuracies]
        net_abrv = path.splitext(f)[0][:4]
        nets_df.loc[len(nets_df)] = [net_abrv, *accuracies]
        net_abbrevs.append(net_abrv)

nets_df.to_csv(path.join(output_folder, "nets.csv"), index=False)


methods = ['bc', 'm1', 'm2']
comb_df = pd.DataFrame(columns=('method', 'topl', *net_abbrevs, *['top' + str(k) for k in computed_accuracies]))
ptrn = r'output_(' + '|'.join([n_abr + "_" for n_abr in net_abbrevs]) + ')+topl_\d+'
print("Processing combin folder {}".format(combin_folder))
for fold in listdir(combin_folder):
    if path.isdir(path.join(combin_folder, fold)) and re.search(ptrn, fold) is not None:
        print("Found combin output {}".format(fold))
        fold_split = fold.split('_')
        topl = int(fold_split[-1])
        cur_nets = fold_split[1:-2]
        for m in methods:
            pred = torch.from_numpy(np.load(path.join(combin_folder, fold, "prob_" + m + ".npy")))
            accuracies = [compute_acc_topk(tar, pred, k) for k in computed_accuracies]
            comb_df.loc[len(comb_df)] = [m, topl, *[1 if net in cur_nets else 0 for net in net_abbrevs], *accuracies]

comb_df.to_csv(path.join(output_folder, "combins.csv"), index=False)



