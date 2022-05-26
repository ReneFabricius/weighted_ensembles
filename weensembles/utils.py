import torch
import gc
import psutil

def gen_probs(c, n, k, device="cpu", dtype=torch.float32):
    """
    Generates random outputs of probabilistic classifiers.

    :param c: number of classifiers
    :param n: number of samples
    :param k: number of classes
    :param device: device on which to output the result
    :param dtype: dtype in which to output the result
    :return: c×n×k tensor of probabilities with last dimension summing to 1
    """
    p = torch.randn(c, n, k, device=device, dtype=dtype)
    p = p - torch.min(p)
    p = p / torch.sum(p, 2).unsqueeze(2)
    return p


def gen_probs_one_source(n, k, device="cpu", dtype=torch.float32):
    """
    Generates random outputs of probabilistic classifier.

    :param n: number of samples
    :param k: number of classes
    :param device: device on which to output the result
    :param dtype: dtype in which to return the result
    :return: n×k tensor of probabilities with last dimension summing to 1
    """
    p = gen_probs(1, n, k, device=device, dtype=dtype)
    p = p.squeeze(dim=0)
    return p


def comp_R(p):
    """
    Computes matrices of pairwise probabilities from classifier predictions
    :param p: classifier predictions
    :return: matrices of pairwise probabilities
    """
    dev = p.device
    dtp = p.dtype
    if len(p.shape) == 2:
        n, k = p.shape
        E = torch.eye(k, device=dev, dtype=dtp)
        VE = (1 - E).unsqueeze(0).expand(n, k, k)
        TCs = VE * p.unsqueeze(2)
    else:
        c, n, k = p.shape
        E = torch.eye(k, device=dev, dtype=dtp)
        VE = (1 - E).unsqueeze(0).unsqueeze(1).expand(c, n, k, k)
        TCs = VE * p.unsqueeze(3)

    R = TCs / (TCs + TCs.transpose(-1, -2) + (TCs == 0))
    return R


def logit(T, eps):
    """
    Computes logit of the input.
    Replaces values below eps by eps and above 1 - eps by 1 - eps before computing the logit.

    :param T: input
    :param eps: epsilon
    :return: logit of the input
    """
    EPS = T.new_full(T.shape, eps, device=T.device, dtype=T.dtype)
    L = torch.where(T < eps, EPS, T)
    LU = torch.where(L > 1 - eps, 1 - EPS, L)
    return torch.log(LU/(1 - LU))


def logit_sparse(T, eps):
    """
    Logit function for sparse tensors.
    Computes logit of the input.
    Replaces values below eps by eps and above 1 - eps by 1 - eps before computing the logit.

    :param T: sparse input
    :param eps: epsilon
    :return: sparse logit of the input
    """
    Vli = logit(T.values(), eps)
    return torch.sparse_coo_tensor(T.indices(), Vli, T.shape)


def pairwise_accuracies(SS, tar):
    """
    Computes accuracy of binary probabilistic classifiers with switched class labels.

    :param SS: c×n×2 tensor, where c is number of classifiers, n is number of samples
    :param tar: correct labels, 0 for class on index 1 and 1 for class on index 0
    :return: accuracies of classifiers
    """
    n = SS.shape[1]
    top_v, top_i = torch.topk(SS, 1, dim=2)
    ti = top_i.squeeze(dim=2)
    # Coding of target is switched. 1 for class on index 0 and 0 for class on index 1
    return torch.sum(ti.cpu() != tar, dim=1)/float(n)


def pairwise_accuracies_penultimate(SS, tar):
    """
        Computes accuracy of difference of supports with switched class labels.

        :param SS: c×n tensor, where c is number of classifiers, n is number of samples,
        positive value means class one, negative class two
        :param tar: correct labels, 1 means class one, 0 means class two
        :return: accuracies of classifiers
        """
    c, n = SS.size()
    ti = SS > 0
    return torch.sum(ti == tar, dim=1) / float(n)


def cuda_mem_try(fun, start_bsz, device, dec_coef=0.5, max_tries=None, verbose=0):
    """Repeatedly to perform action specified by given function which could fail due to cuda oom.
    Each try is performed with lower batch size than previous.

    Args:
        fun (function): Function with one argument - batch size.
        start_bsz (int): STarting batch size.
        dec_coef (float, optional): Coeficient used to multiplicatively decrease batch size. Defaults to 0.5.
        max_tries (int, optional): Maximum number of tries. If None, tries are not limited. Defaults to None.
    """
    batch_size = start_bsz
    while batch_size > 0 and (max_tries is None or max_tries > 0):
        try:
            if verbose > 1:
                print("Trying with batch size {}".format(batch_size))
            return fun(batch_size)
        except RuntimeError as rerr:
            str_err = str(rerr)
            if "memory" not in str_err and "CUDA" not in str_err and "cuda" not in str_err: 
                raise rerr
            if verbose > 1:
                print("CUDA oom exception")
            batch_size = int(dec_coef * batch_size)
            if max_tries is not None:
                max_tries -= 1
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            collected = gc.collect()
            if verbose > 2:
                print(str(rerr))
                print_memory_statistics(device=device, list_tensors=verbose > 3)
                print("Number of garbages collected: {}".format(collected))
            del rerr
    else:
        raise RuntimeError("Unsuccessful to perform the requested action. CUDA out of memory.")
            

def train_test_split_equal_repr(test_size_pc, labels, shuffle=True):
    """Splits data into training/test parts in test part, each class will have equal representation given by
    parameter test_size_pc.

    Args:
        test_size_pc (int): Number of samples per class in test set.
        labels (torch.Tensor): 1 dimensional tensor of class labels
        shuffle (bool, optional): Whether to shuffle data before splitting. Defaults to True.
        :return: (torch.Tensor, torch.Tensor) train indices, test indices.
    """
    
    n = len(labels)
    dev = labels.device
    class_counts = torch.bincount(labels)
    max_count = torch.max(class_counts)
    class_num = class_counts.shape[0]
    if shuffle:
        weights = torch.arange(max_count, device=dev).unsqueeze(0).expand(class_num, max_count)
        weights = (weights < class_counts.unsqueeze(1)).to(dtype=torch.float)
        if torch.min(class_counts) < test_size_pc:
            print("Warning: train_test_split_equal_repr: Not enough samples in each class, picking with replacement.")
            picks = torch.multinomial(weights, test_size_pc, replacement=True)
        else:
            picks = torch.multinomial(weights, test_size_pc, replacement=False)
    else:
        if torch.min(class_counts) < test_size_pc:
            raise ValueError("train_test_split_equal_repr: Not enough samples in each class. Try with shuffle = True - supports picking with replacement")
        picks = torch.arange(test_size_pc, device=dev).unsqueeze(0).expand(class_num, test_size_pc).clone()
        
    vals, sort_perm = torch.sort(labels)
    desort = torch.sort(sort_perm)[1]
    cumsum = torch.cumsum(torch.nn.functional.pad(class_counts[:-1], pad=(1,0), mode="constant", value=0), dim=0)
    picks += cumsum.unsqueeze(1)
    picks = picks.flatten()
    
    pick_map = torch.zeros(n, dtype=torch.bool, device=dev)
    pick_map[picks] = True
    pick_map = pick_map[desort]
    test_indices = sort_perm[picks].squeeze()
    train_indices = torch.nonzero(pick_map != True).squeeze()
    
    return train_indices, test_indices
    
    
def print_memory_statistics(device="gpu:0", list_tensors=False):
    with torch.cuda.device(device):
        allocated = torch.cuda.memory_allocated()
        max_allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_reserved = torch.cuda.max_memory_reserved()

        print("Allocated current: {:.3f}GB, max {:.3f}GB".format(allocated / 2 ** 30, max_allocated / 2 ** 30))
        print("Reserved current: {:.3f}GB, max {:.3f}GB".format(reserved / 2 ** 30, max_reserved / 2 ** 30))

        if list_tensors:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size(), obj.device)
                except:
                    pass
    
    virt_mem = psutil.virtual_memory()
    print("Total: {:.3f}GB, available: {:.3f}GB, used: {:.3f}GB, free: {:.3f}GB".format(
        getattr(virt_mem, "total") / 2 ** 30, getattr(virt_mem, "available") / 2 ** 30, getattr(virt_mem, "used") / 2 ** 30, getattr(virt_mem, "free") / 2 ** 30))