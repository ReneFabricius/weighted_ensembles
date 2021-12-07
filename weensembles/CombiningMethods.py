import torch
from timeit import default_timer as timer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


@torch.no_grad()
def _logreg_sweep_C(X, y, val_X, val_y, fit_intercept=False, verbose=0):
    """
    Performs hyperparameter sweep for regularization parameter C of logistic regression.

    Args:
        X (torch tensor): Tensor of training predictions of combined classifiers, shape number of samples × number of combined classifiers. 
        y (torch tensor): Training labels. 
        val_X (torch tensor): Tensor of validation predicitions of combined classifiers, shape number of samples × number of combined classifiers.
        val_y (torch tensor): Validation labels.
        fit_intercept (bool, optional): Whether to use intercept in the model. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Logistic regression model: Trained logistic regression model with determined C hyperparameter.
    """
    if verbose > 0:
        print("Searching for best C value")
    E_start = -1
    E_end = 1
    E_count = 11
    C_vals = 10**np.linspace(start=E_start, stop=E_end,
                        num=E_count, endpoint=True)
    
    best_C = 1.0
    best_acc = 0.0
    best_model = None
    for C_val in C_vals:
        if verbose > 0:
            print("Testing C value {}".format(C_val))
        clf = LogisticRegression(penalty='l2', fit_intercept=fit_intercept, verbose=max(verbose - 1, 0), C=C_val)
        clf.fit(X.cpu(), y.cpu())
        cur_acc = clf.score(val_X.cpu(), val_y.cpu())
        if verbose > 0:
            print("C value {}, validation accuracy {}".format(C_val, cur_acc))
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_C = C_val
            best_model = clf
    
    if verbose > 0:
        print("C value {} chosen with validation accuracy {}".format(best_C, best_acc))    
        
    return best_model 


@torch.no_grad()
def lda(X, y, verbose=0):
    clf = LinearDiscriminantAnalysis(solver='lsqr')
    clf.fit(X.cpu(), y.cpu())
    
    return clf

setattr(lda, "req_val", False)


@torch.no_grad()
def logreg(X, y, verbose=0):
    clf = LogisticRegression()
    clf.fit(X.cpu(), y.cpu())
    return clf

setattr(logreg, "req_val", False)


@torch.no_grad()
def logreg_no_interc(X, y, verbose=0):
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X.cpu(), y.cpu())
    return clf

setattr(logreg_no_interc, "req_val", False)


@torch.no_grad()
def logreg_sweep_C(X, y, val_X, val_y, verbose=0):
    clf = _logreg_sweep_C(X, y, val_X=val_X, val_y=val_y, fit_intercept=True, verbose=verbose)
    return clf

setattr(logreg_sweep_C, "req_val", True)

    
@torch.no_grad()
def logreg_no_interc_sweep_C(X, y, val_X, val_y, verbose=0):
    clf = _logreg_sweep_C(X, y, val_X=val_X, val_y=val_y, fit_intercept=False, verbose=verbose)
    return clf

setattr(logreg_no_interc_sweep_C, "req_val", True)


comb_methods = {"lda": lda,
                "logreg": logreg,
                "logreg_no_interc": logreg_no_interc,
                "logreg_sweep_C": logreg_sweep_C,
                "logreg_no_interc_sweep_C": logreg_no_interc_sweep_C}


def comb_picker(co_m):
    if co_m not in comb_methods:
        return None 
    
    return comb_methods[co_m]
