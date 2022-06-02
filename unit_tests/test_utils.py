import torch
import numpy as np
import os

def load_npy_arr(file, device, dtype):
    arr = torch.from_numpy(np.load(file)).to(device=torch.device(device), dtype=dtype)
    arr.requires_grad_(False)
    return arr


def load_networks_outputs(nn_outputs_path, experiment_out_path=None, device='cpu', dtype=torch.float, load_train_data=True):
    """
    Loads network outputs for single replication. Dimensions in the output tensors are network, sample, class.
    :param nn_outputs_path: replication outputs path.
    :param experiment_out_path: if not None a path to folder where to store networks_order file
    containing the order of the networks
    :param device: device to use
    :return: dictionary with network outputs and labels
    """
    networks = [fold for fold in os.listdir(nn_outputs_path) if os.path.isdir(os.path.join(nn_outputs_path, fold))]

    if experiment_out_path is not None:
        networks_order = open(os.path.join(experiment_out_path, 'networks_order.txt'), 'w')
        for net in networks:
            networks_order.write(net + "\n")
        networks_order.close()

    
    test_outputs = []
    for net in networks:
        test_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'test_outputs.npy'), device=device, dtype=dtype).
                            unsqueeze(0))
    test_outputs = torch.cat(test_outputs, 0)
    test_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'test_labels.npy'), device=device, dtype=torch.long)

    if load_train_data:
        train_outputs = []
        for net in networks:
            train_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'train_outputs.npy'), device=device, dtype=dtype).
                                unsqueeze(0))
        train_outputs = torch.cat(train_outputs, 0)
        train_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'train_labels.npy'), device=device, dtype=torch.long)

    val_outputs = []
    for net in networks:
        val_outputs.append(load_npy_arr(os.path.join(nn_outputs_path, net, 'val_outputs.npy'), device=device, dtype=dtype).
                           unsqueeze(0))
    val_outputs = torch.cat(val_outputs, 0)
    val_labels = load_npy_arr(os.path.join(nn_outputs_path, networks[0], 'val_labels.npy'), device=device, dtype=torch.long)

    ret = {"val_outputs": val_outputs,
            "val_labels": val_labels, "test_outputs": test_outputs, "test_labels": test_labels,
            "networks": networks}
    if load_train_data:
        ret["train_outputs"] = train_outputs
        ret["train_labels"] = train_labels
    
    return ret
