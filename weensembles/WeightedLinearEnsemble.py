import numpy as np
import torch
import pickle
import pandas as pd
from scipy.stats import normaltest
from timeit import default_timer as timer
from torch._C import device
from torch.special import expit

from weensembles.CouplingMethods import coup_picker
from weensembles.CombiningMethods import comb_picker
from weensembles.utils import logit, pairwise_accuracies, pairwise_accuracies_penultimate
from weensembles.Ensemble import Ensemble


class WeightedLinearEnsemble(Ensemble):
    def __init__(self, c=0, k=0, device=torch.device("cpu"), dtp=torch.float32):
        """
        Trainable ensembling of classification posteriors.
        :param c: Number of classifiers
        :param k: Number of classes
        :param device: Torch device to use for computations
        :param dtp: Torch datatype to use for computations
        """
        super().__init__(c=c, k=k, device=device, dtp=dtp)
        self.logit_eps_ = 1e-5
        self.comb_model_ = None 

    def fit(self, preds, labels, combining_method,
            verbose=0, val_preds=None, val_labels=None, **kwargs):
        """
        Trains combining method on logits of several classifiers.
        
        Args:
            combining_method (string): Combining method to use.
            preds (torch.tensor): c x n x k tensor of constituent classifiers outputs.
            c - number of constituent classifiers, n - number of training samples, k - number of classes
            labels (torch.tensor): n tensor of sample labels
            verbose (int): Verbosity level.
            val_preds (torch.tensor): Validation set used for hyperparameter sweep. Required if combining_method.req_val is True.
            val_labels (torch.tensor): Validation set targets. Required if combining_method.req_val is True. 
        """
        if verbose > 0:
            print("Starting fit, combining method: {}".format(combining_method))
        comb_m = comb_picker(combining_method, c=self.c_, k=self.k_, device=self.dev_, dtype=self.dtp_)
        if comb_m is None:
            raise ValueError("Unknown combining method {} selected".format(combining_method))
        
        self.comb_model_ = comb_m
        
        inc_val = comb_m.req_val_
        if inc_val and (val_preds is None or val_labels is None):
            raise ValueError("val_preds and val_labels are required for combining method {}".format(combining_method))
        
        self.comb_model_.fit(X=preds, y=labels, val_X=val_preds, val_y=val_labels, verbose=verbose, **kwargs)
        
            
    @torch.no_grad()
    def predict_proba(self, preds, coupling_method, verbose=0, l=None, batch_size=None, predict_uncertainty=False):   
        """
        Combines outputs of constituent classifiers using all classes.
        
        Args:
            batch_size (int, optional): batch size for coupling method, default None - single batch. Defaults to None.
            verbosity (int, optional): Level of detailed output. Defaults to 0.
            preds (torch.tensor): c x n x k tensor of constituent classifiers posteriors
            c - number of constituent classifiers, n - number of training samples, k - number of classes
            coupling_method (str): coupling method to use
            l (int, optional): If specified, only top l classes of each classifier are considered in the final prediction. Defaults to None.
            predict_uncertainty(bool, optional): Whether to compute uncertainty measure. Defaults to False.
            
        Returns: 
            torch.tensor: n x k tensor of combined posteriors
        """
        probs = self.comb_model_.predict_proba(X=preds, coupling_method=coupling_method, l=l, verbose=verbose,
                                               batch_size=batch_size, predict_uncertainty=predict_uncertainty)

        return probs

    @torch.no_grad()
    def save(self, file, verbose=0):
        """
        Save trained ensemble into a file.
        :param file: file to save the models to
        :return:
        """
        if verbose > 0:
            print("Saving ensemble into file: " + str(file))
        if self.comb_model_ is not None:
            self.comb_model_.to_cpu()

        with open(file, 'wb') as f:
            pickle.dump(self.__dict__, f)
            
        if self.comb_model_ is not None:
            self.comb_model_.to_dev()

    @torch.no_grad()
    def load(self, file, verbose=0):
        """
        Load trained ensemble from a file.
        :param file: File to load the ensemble from.
        :return:
        """
        if verbose > 0:
            print("Loading models from file: " + str(file))
        with open(file, 'rb') as f:
            dump_dict = pickle.load(f)

        keep_dev = self.dev_
        self.__dict__.update(dump_dict)
        self.dev_ = keep_dev
        if self.comb_model_ is not None:
            self.comb_model_.set_dev(self.dev_)
            self.comb_model_.to_dev()

    @torch.no_grad()
    def save_coefs_csv(self, file):
        """
        Save linear coefficients into a csv file.
        :param file: file to save the coefficients to
        :return:
        """
        Ls = [None] * ((self.k_ * (self.k_ - 1)) // 2)
        li = 0
        cols = ["i", "j"] + ["coef" + str(k) for k in range(self.c_)] + ["interc"]
        for i in range(self.k_):
            for j in range(i + 1, self.k_):
                cfs = [[i, j] + self.comb_model_.coefs_[i, j].tolist()]
                Ls[li] = pd.DataFrame(cfs, columns=cols)
                li += 1

        df = pd.concat(Ls, ignore_index=True)
        df.to_csv(file, index=False)

    @torch.no_grad()
    def save_C_coefs(self, file):
        """Method usable for logreg configurations with sweep_C option and save_C parameter during fit.
        Saves best found regularization coefficients C into a specified file.

        Args:
            file (_type_): Path to the file to save coefficients to.
        """
        if not hasattr(self.comb_model_, "best_C_"):
            print("Warning: combining method {} does not have best_C_ attribute".format(self.comb_model_.__name__))
            return
        
        Ls = [None] * ((self.k_ * (self.k_ - 1)) // 2)
        li = 0
        cols = ["i", "j", "C"]
        for i in range(self.k_):
            for j in range(i + 1, self.k_):
                cfs = [[i, j, self.comb_model_.best_C_[i, j].item()]]
                Ls[li] = pd.DataFrame(cfs, columns=cols)
                li += 1

        df = pd.concat(Ls, ignore_index=True)
        df.to_csv(file, index=False)
