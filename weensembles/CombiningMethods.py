import torch
from timeit import default_timer as timer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_array
from scipy.special import expit
import numpy as np

from weensembles.CalibrationMethod import TemperatureScaling

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
setattr(lda, "fit_pairwise", True)


@torch.no_grad()
def logreg(X, y, verbose=0):
    clf = LogisticRegression()
    clf.fit(X.cpu(), y.cpu())
    return clf

setattr(logreg, "req_val", False)
setattr(logreg, "fit_pairwise", True)

@torch.no_grad()
def logreg_no_interc(X, y, verbose=0):
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X.cpu(), y.cpu())
    return clf

setattr(logreg_no_interc, "req_val", False)
setattr(logreg_no_interc, "fit_pairwise", True)


@torch.no_grad()
def logreg_sweep_C(X, y, val_X, val_y, verbose=0):
    clf = _logreg_sweep_C(X, y, val_X=val_X, val_y=val_y, fit_intercept=True, verbose=verbose)
    return clf

setattr(logreg_sweep_C, "req_val", True)
setattr(logreg_sweep_C, "fit_pairwise", True)

    
@torch.no_grad()
def logreg_no_interc_sweep_C(X, y, val_X, val_y, verbose=0):
    clf = _logreg_sweep_C(X, y, val_X=val_X, val_y=val_y, fit_intercept=False, verbose=verbose)
    return clf

setattr(logreg_no_interc_sweep_C, "req_val", True)
setattr(logreg_no_interc_sweep_C, "fit_pairwise", True)


class Averager():
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.penultimate_ = True
    
    def fit(self, X, y, calibrate=False, verbose=0):
        c, n, k = X.shape
        if not calibrate:
            self.coef_ = np.full(shape=(1, c), fill_value=1.0 / c)
        else:
            self.coef_ = np.zeros(shape=(1, c))
            for ci in range(c):
                ts = TemperatureScaling(device=X.device, dtp=X.dtype)
                ts.fit(X[ci], y, verbose=verbose)
                self.coef_[0, ci] = 1.0 / ts.temp_.item()
            
        self.intercept_ = np.zeros(shape=(1))
    
    def decision_function(self, X):
        X = check_array(X)
        return np.squeeze(X @ self.coef_.T)
    
    def predict_proba(self, X):
        prob = self.decision_function(X)
        expit(prob, out=prob)
        return np.vstack([1 - prob, prob]).T
    
    def score(self, X, y):
        prob = self.predict_proba(X)
        preds = np.argmax(prob, axis=1)
        return np.sum(preds == y) / len(y)
        
        
@torch.no_grad()
def average(X, y, verbose=0):
    clf = Averager()
    clf.fit(X, y)
    return clf

setattr(average, "req_val", False)
setattr(average, "fit_pairwise", False)


@torch.no_grad()
def cal_average(X, y, val_X, val_y, verbose=0):
    clf = Averager()
    clf.fit(val_X, val_y, calibrate=True)
    return clf
 
setattr(cal_average, "req_val", True)
setattr(cal_average, "fit_pairwise", False)

   

comb_methods = {"lda": lda,
                "logreg": logreg,
                "logreg_no_interc": logreg_no_interc,
                "logreg_sweep_C": logreg_sweep_C,
                "logreg_no_interc_sweep_C": logreg_no_interc_sweep_C,
                "average": average,
                "cal_average": cal_average}


def comb_picker(co_m):
    if co_m not in comb_methods:
        return None 
    
    return comb_methods[co_m]
