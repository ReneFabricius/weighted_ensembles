import torch
from timeit import default_timer as timer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_array
from scipy.special import expit
import numpy as np
from torch.optim import optimizer

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


class Averager():
    def __init__(self, combine_probs = False, coefs=None, intercept=None):
        """Initializes averager combining model.

        Args:
            combine_probs (bool, optional): If true, multiplies provided inputs by coefficients, computed expits and then averages results. Defaults to False.
        """
        self.coef_ = coefs 
        self.intercept_ = intercept 
        self.penultimate_ = True
        self.combine_probs_ = combine_probs
    
    def fit(self, X, y, calibrate=False, verbose=0):
        """Fits the averager combining model.

        Args:
            X (tensor): Tensor of training predictors. Shape c × n × k, where c is number of classifiers, n is number of training samples ank k in number of classes.
            y (tensor): Tensor of traininng labels. Shape n.
            calibrate (bool, optional): Whether to set coefficients equal to inverse of temperature scaling temperatures. Defaults to False.
            verbose (int, optional): [description]. Defaults to 0.
        """
        c, n, k = X.shape
        if not calibrate:
            if not self.combine_probs_:
                self.coef_ = np.full(shape=(1, c), fill_value=1.0 / c)
            else:
                self.coef_ = np.ones(shape=(1, c))
        else:
            self.coef_ = np.zeros(shape=(1, c))
            for ci in range(c):
                ts = TemperatureScaling(device=X.device, dtp=X.dtype)
                ts.fit(X[ci], y, verbose=verbose)
                self.coef_[0, ci] = 1.0 / ts.temp_.item()
            
        self.intercept_ = np.zeros(shape=(1))
    
    def decision_function(self, X):
        X = check_array(X)
        return np.squeeze(X @ self.coef_.T + self.intercept_)
    
    def predict_proba(self, X):
        """Combines inputs and predicts probability.

        Args:
            X (tensor): Tensor of inputs. Shape n × c.

        Returns:
            [tensor]: Resulting probabilities. Shape n × 2. Probability of class for which supports are given is at index 1. 
        """
        if not self.combine_probs_:
            prob = self.decision_function(X)
            expit(prob, out=prob)
        else:
            X = check_array(X)
            sup = X * self.coef_
            probs = expit(sup)
            prob = np.mean(probs, axis=-1)
        
        return np.vstack([1 - prob, prob]).T
    
    def score(self, X, y):
        prob = self.predict_proba(X)
        preds = np.argmax(prob, axis=1)
        return np.sum(preds == y) / len(y)
 

def grad_comb(X, y, wle, coupling_method, verbose=0):
    if verbose > 0:
        print("Starting grad_m1 fit")
    c, n, k = X.shape
    epochs = 10
    batch_sz = 500
    lr = 0.01
    momentum = 0.9
    
    wle.trained_on_penultimate_ = True
    coefs = torch.full(size=(k, k, c + 1), fill_value=1.0 / c, device=X.device, dtype=X.dtype)
    coefs[:, :, c] = 0
    coefs.requires_grad_(True)
    X.requires_grad_(False)
    y.requires_grad_(False)
    nll_loss = torch.nn.NLLLoss()
    opt = torch.optim.SGD(params=(coefs,), lr=lr, momentum=momentum)
    
    for e in range(epochs):
        if verbose > 1:
            print("Processing epoch {} out of {}".format(e, epochs))
        perm = torch.randperm(n=n, device=X.device)
        X_perm = X[:, perm]
        y_perm = y[perm]
        for batch_s in range(0, n, batch_sz):
            X_batch = X_perm[:, batch_s:(batch_s + batch_sz)]
            y_batch = y_perm[batch_s:(batch_s + batch_sz)]
            if verbose > 1:
                print("Epoch {}: [{}/{}]".format(e, batch_s + len(y_batch), n))

            mbatch_sz = batch_sz
            finished = False
            
            while not finished and mbatch_sz > 0:
                try:
                    opt.zero_grad()
                    if verbose > 1:
                        print("Trying micro batch size {}".format(mbatch_sz))
                    for mbatch_s in range(0, len(y_batch), mbatch_sz):
                        X_mb = X_batch[:, mbatch_s:(mbatch_s + mbatch_sz)]
                        y_mb = y_batch[mbatch_s:(mbatch_s + mbatch_sz)]
                        pred = wle.predict_proba_topl_fast(MP=X_mb, l=k, coupling_method="m1", coefs=coefs, verbose=verbose)
                        loss = nll_loss(pred, y_mb) * (len(y_mb) / len(y_batch))
                        loss.backward()
                    
                    opt.step()
                    finished = True

                except RuntimeError as rerr:
                    if 'memory' not in str(rerr):
                        raise rerr
                    if verbose > 1:
                        print("OOM Exception")
                    del rerr
                    mbatch_sz = int(0.5 * mbatch_sz)
                    torch.cuda.empty_cache()

    avgs = [[None for _ in range(k)] for _ in range(k)]
    coefs.requires_grad_(False)
    for fc in range(k):
        for sc in range(fc + 1, k):
            avgs[fc][sc] = Averager(combine_probs=False, coefs=coefs[fc, sc, 0:c], intercept=coefs[fc, sc, [c]])
    
    return avgs
 

@torch.no_grad()
def lda(X, y, wle, verbose=0):
    clf = LinearDiscriminantAnalysis(solver='lsqr')
    clf.fit(X.cpu(), y.cpu())
    
    return clf

setattr(lda, "req_val", False)
setattr(lda, "fit_pairwise", True)
setattr(lda, "combine_probs", False)


@torch.no_grad()
def logreg(X, y, wle, verbose=0):
    clf = LogisticRegression()
    clf.fit(X.cpu(), y.cpu())
    return clf

setattr(logreg, "req_val", False)
setattr(logreg, "fit_pairwise", True)
setattr(logreg, "combine_probs", False)

@torch.no_grad()
def logreg_no_interc(X, y, wle, verbose=0):
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X.cpu(), y.cpu())
    return clf

setattr(logreg_no_interc, "req_val", False)
setattr(logreg_no_interc, "fit_pairwise", True)
setattr(logreg_no_interc, "combine_probs", False)


@torch.no_grad()
def logreg_sweep_C(X, y, val_X, val_y, wle, verbose=0):
    clf = _logreg_sweep_C(X, y, val_X=val_X, val_y=val_y, fit_intercept=True, verbose=verbose)
    return clf

setattr(logreg_sweep_C, "req_val", True)
setattr(logreg_sweep_C, "fit_pairwise", True)
setattr(logreg_sweep_C, "combine_probs", False)

    
@torch.no_grad()
def logreg_no_interc_sweep_C(X, y, val_X, val_y, wle, verbose=0):
    clf = _logreg_sweep_C(X, y, val_X=val_X, val_y=val_y, fit_intercept=False, verbose=verbose)
    return clf

setattr(logreg_no_interc_sweep_C, "req_val", True)
setattr(logreg_no_interc_sweep_C, "fit_pairwise", True)
setattr(logreg_no_interc_sweep_C, "combine_probs", False)
       
        
@torch.no_grad()
def average(X, y, wle, verbose=0):
    clf = Averager()
    clf.fit(X, y, verbose=verbose)
    return clf

setattr(average, "req_val", False)
setattr(average, "fit_pairwise", False)
setattr(average, "combine_probs", False)


@torch.no_grad()
def cal_average(X, y, val_X, val_y, wle, verbose=0):
    clf = Averager()
    clf.fit(val_X, val_y, calibrate=True, verbose=verbose)
    return clf
 
setattr(cal_average, "req_val", True)
setattr(cal_average, "fit_pairwise", False)
setattr(cal_average, "combine_probs", False)


@torch.no_grad()
def prob_average(X, y, wle, verbose=0):
    clf = Averager(combine_probs=True)
    clf.fit(X, y, verbose=verbose)
    return clf

setattr(prob_average, "req_val", False)
setattr(prob_average, "fit_pairwise", False)
setattr(prob_average, "combine_probs", True)


@torch.no_grad()
def cal_prob_average(X, y, val_X, val_y, wle, verbose=0):
    clf = Averager(combine_probs=True)
    clf.fit(val_X, val_y, calibrate=True, verbose=verbose)
    return clf

setattr(cal_prob_average, "req_val", True)
setattr(cal_prob_average, "fit_pairwise", False)
setattr(cal_prob_average, "combine_probs", True)


def grad_m1(X, y, wle, verbose=0):
    return grad_comb(X=X, y=y, wle=wle, coupling_method="m1", verbose=verbose)
                       
setattr(grad_m1, "req_val", False)
setattr(grad_m1, "fit_pairwise", False)
setattr(grad_m1, "combine_probs", False)


def grad_m2(X, y, wle, verbose=0):
    return grad_comb(X=X, y=y, wle=wle, coupling_method="m2", verbose=verbose)
                       
setattr(grad_m2, "req_val", False)
setattr(grad_m2, "fit_pairwise", False)
setattr(grad_m2, "combine_probs", False)


def grad_bc(X, y, wle, verbose=0):
    return grad_comb(X=X, y=y, wle=wle, coupling_method="bc", verbose=verbose)
                       
setattr(grad_bc, "req_val", False)
setattr(grad_bc, "fit_pairwise", False)
setattr(grad_bc, "combine_probs", False)

                      
comb_methods = {"lda": lda,
                "logreg": logreg,
                "logreg_no_interc": logreg_no_interc,
                "logreg_sweep_C": logreg_sweep_C,
                "logreg_no_interc_sweep_C": logreg_no_interc_sweep_C,
                "average": average,
                "cal_average": cal_average,
                "prob_average": prob_average,
                "cal_prob_average": cal_prob_average,
                "grad_m1": grad_m1,
                "grad_m2": grad_m2,
                "grad_bc": grad_bc}


def comb_picker(co_m):
    if co_m not in comb_methods:
        return None 
    
    return comb_methods[co_m]
