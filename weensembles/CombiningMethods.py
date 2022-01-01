import re
import torch
from timeit import default_timer as timer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_array
from torch.special import expit
import numpy as np
from abc import ABC, abstractmethod

from weensembles.CalibrationMethod import CalibrationMethod, TemperatureScaling
from weensembles.CouplingMethods import coup_picker
from weensembles.predictions_evaluation import compute_acc_topk, compute_nll, compute_pairwise_accuracies
from weensembles.utils import cuda_mem_try, pairwise_accuracies

class GeneralCombiner(ABC):
    """Abstract class for combining methods.

    Raises:
        rerr: [description]

    Returns:
        GeneralCombiner: Combining method. Doesn't hold coefficients.
    """
    
    def __init__(self, c, k, req_val=False, fit_pairwise=False, combine_probs=False, device="cpu", dtype=torch.float, name="no_name"):
        self.req_val_ = req_val
        self.fit_pairwise_ = fit_pairwise
        self.combine_probs_ = combine_probs
        self.dev_ = device
        self.dtp_ = dtype
        self.name_ = name
        self.coefs_ = None
        self.c_ = c
        self.k_ = k
        
    def fit(self, X, y, val_X=None, val_y=None, verbose=0):
        """
        Trains combining method on logits of several classifiers.
        
        Args:
            combining_method (string): Combining method to use.
            X (torch.tensor): c x n x k tensor of constituent classifiers outputs.
            c - number of constituent classifiers, n - number of training samples, k - number of classes
            y (torch.tensor): n tensor of sample labels
            verbose (int): Verbosity level.
            val_X (torch.tensor): Validation set used for hyperparameter sweep. Required if combining_method.req_val is True.
            val_y (torch.tensor): Validation set targets. Required if combining_method.req_val is True. 
        """

        start = timer()
        num = self.k_ * (self.k_ - 1) // 2      # Number of pairs of classes
        print_step = num // 100

        if self.fit_pairwise_:
            self.coefs_ = torch.zeros(self.k_, self.k_, self.c_ + 1, device=self.dev_, dtype=self.dtp_)
            pi = 0
            for fc in range(self.k_):
                for sc in range(fc + 1, self.k_):
                    if print_step > 0 and pi % print_step == 0:
                        print("Fit progress {}%".format(pi // print_step), end="\r")

                    # c x n tensor containing True for samples belonging to classes fc, sc
                    SamM = (y == fc) + (y == sc)
                    if self.req_val_:
                        SamM_val = (val_y == fc) + (val_y == sc)
                        
                    # c x s x 1 tensor, where s is number of samples in classes fc and sc.
                    # Tensor contains support of networks for class fc minus support for class sc
                    SS = X[:, SamM][:, :, fc] - X[:, SamM][:, :, sc]

                    # s x c tensor of logit supports of k networks for class fc against class sc for s samples
                    pw_X = SS.squeeze().transpose(0, 1)

                    if self.req_val_:
                        SS_val = val_X[:, SamM_val][:, :, fc] - val_X[:, SamM_val][:, :, sc]
                        pw_X_val = SS_val.squeeze().transpose(0, 1)
                           
                    # Prepare targets
                    pw_y = y[SamM]
                    mask_fc = (pw_y == fc)
                    mask_sc = (pw_y == sc)
                    pw_y[mask_fc] = 1
                    pw_y[mask_sc] = 0
                    
                    if self.req_val_:
                        pw_y_val = val_y[SamM_val]
                        mask_fc_val = (pw_y_val == fc)
                        mask_sc_val = (pw_y_val == sc)
                        pw_y_val[mask_fc_val] = 1
                        pw_y_val[mask_sc_val] = 0

                    if self.req_val_:
                        pw_coefs = self.train(X=pw_X, y=pw_y, val_X=pw_X_val, val_y=pw_y_val, verbose=verbose)
                    else:
                        pw_coefs = self.train(X=pw_X, y=pw_y, verbose=verbose)
                        
                    self.coefs_[fc, sc, :] = pw_coefs
                    self.coefs_[sc, fc, :] = pw_coefs

                    if verbose > 1:
                        pwacc = pairwise_accuracies(SS, pw_y)
                        print("Training pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                                "\n\tpairwise accuracies: " + str(pwacc) +
                                "\n\tchosen coefficients: " + str(pw_coefs[:, 0:-1]) +
                                "\n\tintercept: " + str(pw_coefs[:, -1]))

                        print("\tcombined accuracy: " + str(self.score(X=X, y=y, coefs=pw_coefs)))

                    pi += 1

        else:
            if self.req_val_:
                self.coefs_ = self.train(X=X, y=y, val_X=val_X, val_y=val_y, verbose=verbose)
            else:
                self.coefs_ = self.train(X=X, y=y, val_X=None, val_y=None, verbose=verbose)
                
        end = timer()
        if verbose > 0:
            print("Fit finished in " + str(end - start) + " s")
        
    
    @abstractmethod
    def train(self, X, y, val_X=None, val_y=None, verbose=0):
        pass
    
    def predict_proba(self, X, coupling_method, l=None, verbose=0, batch_size=None, coefs=None, combine_probs=None):
        """
        Combines outputs of constituent classifiers using only those classes, which are among the top l most probable
        for some constituent classifier.
        :param MP: MP: c x n x k tensor of constituent classifiers posteriors
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param l: how many most probable classes for each constituent classifier to consider
        :param coupling_method: coupling method to use
        :param batch_size: if not none, the size of the batches to use
        :return: n x k tensor of combined posteriors
        """
        if verbose > 0:
            print("Starting predict proba of {}".format(self.name_))
        coup_m = coup_picker(coupling_method)
        if coup_m is None:
            print("Unknown coupling method {} selected".format(coupling_method))
            return 1
 
        if self.coefs_ is None and coefs is None: 
            print("Ensemble not trained")
            return
        start = timer()
        c, n, k = X.size()
        assert c == self.c_
        assert k == self.k_

        if l is None:
            l = self.k_
        b_size = batch_size if batch_size is not None else n
        ps_list = []

        for start_ind in range(0, n, b_size):
            curMP = X[:, start_ind:(start_ind + b_size), :].to(device=self.dev_, dtype=self.dtp_)
            curn = curMP.shape[1]

            # ind is c x n x l tensor of top l indices for each sample in each network output
            val, ind = torch.topk(curMP, l, dim=2)
            M = torch.zeros(curMP.shape, dtype=torch.bool, device=self.dev_)
            # place true in positions of top probs
            # c x n x k tensor
            M.scatter_(2, ind, True)
            # combine selections over c inputs
            # n x k tensor containing for each sample a mask of union of top l classes from each constituent classifier
            M = torch.sum(M, dim=0, dtype=torch.bool)
            ps = torch.zeros(curn, k, device=self.dev_, dtype=self.dtp_)
            # Selected class counts for every n
            NPC = torch.sum(M, 1).squeeze()
            # goes over possible numbers of classes in union of top l classes from each constituent classifier
            for pc in range(l, l * c + 1):
                # Pick those samples which have pc classes in the union
                # pcn x c x k tensor
                pcMP = curMP[:, NPC == pc]
                # Pick pc-class masks
                # pcn x k tensor
                pcM = M[NPC == pc]
                # Number of samples with pc classes in the union
                pcn = pcM.shape[0]
                if pcn == 0:
                    continue
                
                # pcn x pc x c tensor of picked supports
                picked_supports = pcMP[:, pcM].reshape(c, pcn, pc).transpose(0, 1).transpose(1, 2)
                exp_supports = picked_supports.unsqueeze(2).expand(pcn, pc, pc, c)
                # pcn x pc x pc x c tensor of pairwise supports
                pw_supports = exp_supports - exp_supports.transpose(1, 2)
                
                pc_coef_M = pcM.unsqueeze(2).expand(pcn, k, k) * pcM.unsqueeze(1).expand(pcn, k, k)
                if coefs is None:
                    pc_coefs = self.coefs_.unsqueeze(0).expand(pcn, k, k, c + 1)[pc_coef_M, :].reshape(pcn, pc, pc, c + 1)
                else:
                    pc_coefs = coefs.unsqueeze(0).expand(pcn, k, k, c + 1)[pc_coef_M, :].reshape(pcn, pc, pc, c + 1)
                
                Ws = pc_coefs[:, :, :, 0:-1]
                Bs = pc_coefs[:, :, :, -1]
                
                pw_w_supports = pw_supports * Ws
                if (combine_probs is None and self.combine_probs_) or combine_probs:
                    pw_probs = expit(pw_w_supports)
                    pcR = torch.mean(pw_probs, dim=-1)
                else:
                    pcR = expit(torch.sum(pw_w_supports, dim=-1) + Bs)

                pc_probs = coup_m(pcR, verbose=verbose)
                pc_ps = torch.zeros(pcn, k, device=self.dev_, dtype=self.dtp_)
                pc_ps[pcM] = torch.flatten(pc_probs)
                ps[NPC == pc] = pc_ps
            
            ps_list.append(ps)

        ps_full = torch.cat(ps_list, dim=0)
        end = timer()
        if verbose > 0:
            print("Predict proba topl fast simple finished in " + str(end - start) + " s")

        return ps_full


    def score(self, X, y, coefs):
        """Computes accuracy of model with given coefficients on given data.

        Args:
            X (torch.tensor): Tensor of testing predictions. Shape n × c. Where c is number of combined classifiers and n is number of samples.
            y (torch.tensor): Tensor of correct classes. Shape n - number of samples.
            coefs (torch.tensor): Tensor of model coefficients. Shape 1 × (c + 1). Where c is number of combined classifiers.

        Returns:
            float: Top 1 accuracy of model.
        """
        prob = self.predict_proba(X=X, coefs=coefs)
        preds = torch.argmax(prob, dim=1)
        return torch.sum(preds == y).item() / len(y)
    

@torch.no_grad()
def _logreg_sweep_C(X, y, val_X, val_y, fit_intercept=False, verbose=0, device="cpu", dtype=torch.float):
    """
    Performs hyperparameter sweep for regularization parameter C of logistic regression.

    Args:
        X (torch.tensor): Tensor of training predictions of combined classifiers, shape number of samples × number of combined classifiers. 
        y (torch.tensor): Training labels. 
        val_X (torch.tensor): Tensor of validation predicitions of combined classifiers, shape number of samples × number of combined classifiers.
        val_y (torch.tensor): Validation labels.
        fit_intercept (bool, optional): Whether to use intercept in the model. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        torch.tensor: Coefficients of trained logistic regression model with determined C hyperparameter.
    """
    if verbose > 1:
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
        if verbose > 1:
            print("Testing C value {}".format(C_val))
        clf = LogisticRegression(penalty='l2', fit_intercept=fit_intercept, verbose=max(verbose - 1, 0), C=C_val)
        clf.fit(X.cpu(), y.cpu())
        cur_acc = clf.score(val_X.cpu(), val_y.cpu())
        if verbose > 1:
            print("C value {}, validation accuracy {}".format(C_val, cur_acc))
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_C = C_val
            best_model = clf
    
    if verbose > 1:
        print("C value {} chosen with validation accuracy {}".format(best_C, best_acc))    
        
    coefs = torch.cat((torch.tensor(best_model.coef_, device=device, dtype=dtype).squeeze(),
                       torch.tensor(best_model.intercept_, device=device, dtype=dtype)))

    return coefs 


def _averaging_coefs(X, y, val_X=None, val_y=None, calibrate=False, comb_probs=False, device="cpu", dtype=torch.float, verbose=0):
    """Computes coefficients for averaging family of combining methods.

    Args:
        X (torch.tensor): Tensor of training predictors. Shape c × n × k. Where c is number of combined classifiers, 
        n is number of trainign samples and k is number of classes.
        y (torch.tensor): Tensor of training labels. Shape n - number of training samples.
        val_X (torch.tensor, optional): Tensor of validation predictors. Shape c × n × k. Where c is number of combined classifiers, 
        n is number of validation samples and k is number of classes Defaults to None.
        val_y (torch.tensor, optional): Tensor of validation labels. Shape n - number of validation samples. Defaults to None.
        calibrate (bool, optional): Whether to perform calibrated average. Defaults to False.
        comb_probs (bool, optional): Whether to combine probabilities, not logits. Defaults to False.
        device (str, optional): Device of which to perform computations and return coefficients. Defaults to "cpu".
        dtype (torch.dtype, optional): Requested dtype of coefficients and computations. Defaults to torch.float.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        torch.tensor: Tensor of combining coefficients. Shape k × k × (c + 1), where k is number of classes and c is number of combined classifiers.
    """
    if calibrate:
        c, n, k = val_X.shape
        coefs = torch.zeros(size=(c + 1, ), device=device, dtype=dtype)
        for ci in range(c):
            ts = TemperatureScaling(device=device, dtp=dtype)
            ts.fit(val_X[ci], val_y, verbose=verbose)
            coefs[ci] = 1.0 / ts.temp_.item()

    else:
        c, n, k = X.shape
        if comb_probs:
            coefs = torch.ones(size=(c + 1, ), device=device, dtype=dtype)
        else:
            coefs = torch.full(size=(c + 1, ), fill_value=1.0 / c, device=device, dtype=dtype)
        
        coefs[c] = 0
    
    return coefs.expand(k, k, -1)


def _grad_comb(X, y, combiner, coupling_method, verbose=0, epochs=10, lr=0.3, momentum=0.85, test_period=None, batch_sz=500):
    """Trains combining coefficients in end-to-end manner by gradient descent method.

    Args:
        X (torch.tensor): Tensor of training predictors. Shape c × n × k. Where c is number of combined classifiers, n is number of training samples and k is number of classes.
        y (torch.tensor): Tensor of training labels. Shape n - number of training samples.
        wle (WeightedLinearEnsemble): WeightedLinearEnsemble model for which the coefficients are trained. Prediction method of this model is needed.
        coupling_method (str): Name of coupling method to be used for training.
        verbose (int, optional): Level of verbosity. Defaults to 0.
        epochs (int, optional): Number of epochs. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 0.01.
        momentum (float, optional): Momentum. Defaults to 0.9.
        test_period (int, optional): If not None, period in which testing pass is performed. Defaults to None.
        batch_sz (int, optional): Batch size. Defaults to 500.

    Raises:
        rerr: Possible cuda memory error.

    Returns:
        torch.tensor: Tensor of model coefficients. Shape k × k × (c + 1). Where k is number of classes and c is number of combined classifiers.
    """
    if verbose > 0:
        print("Starting grad_{} fit".format(coupling_method))
    c, n, k = X.shape
    
    coefs = torch.full(size=(k, k, c + 1), fill_value=1.0 / c, device=X.device, dtype=X.dtype)
    coefs[:, :, c] = 0
    coefs.requires_grad_(True)
    X.requires_grad_(False)
    y.requires_grad_(False)
    nll_loss = torch.nn.NLLLoss()
    opt = torch.optim.SGD(params=(coefs,), lr=lr, momentum=momentum)
    thr_value = 1e-9
    thresh = torch.nn.Threshold(threshold=thr_value, value=thr_value)
    
    if test_period is not None:
        with torch.no_grad():
            test_bsz = X.shape[1]
            test_pred = cuda_mem_try(
                fun=lambda bsz: combiner.predict_proba(X=X, l=k, coupling_method=coupling_method, verbose=max(verbose - 2, 0), batch_size=bsz, coefs=coefs),
                start_bsz=test_bsz, verbose=verbose)
            
            acc = compute_acc_topk(pred=test_pred, tar=y, k=1)
            nll = compute_nll(pred=test_pred, tar=y)
            print("Before training: acc {}, nll {}".format(acc, nll))

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
                        pred = thresh(combiner.predict_proba(X=X_mb, l=k, coupling_method=coupling_method,
                                                             verbose=max(verbose - 2, 0), batch_size=mbatch_sz, coefs=coefs))
                        loss = nll_loss(torch.log(pred), y_mb) * (len(y_mb) / len(y_batch))
                        if verbose > 1:
                            print("Loss: {}".format(loss))
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
            
        if test_period is not None and (e + 1) % test_period == 0:
            with torch.no_grad():
                test_pred = cuda_mem_try(
                    fun=lambda bsz: combiner.predict_proba(X=X, l=k, coupling_method=coupling_method, verbose=max(verbose - 2, 0), batch_size=bsz, coefs=coefs),
                    start_bsz=test_bsz, verbose=verbose)

                acc = compute_acc_topk(pred=test_pred, tar=y, k=1)
                nll = compute_nll(pred=test_pred, tar=y)
                print("Test epoch {}: acc {}, nll {}".format(e, acc, nll))

    coefs.requires_grad_(False)
    return coefs
 

class lda(GeneralCombiner):
    """Combining method which uses Linear DIscriminant Analysis to infer combining coefficients.
    """
    def __init__(self, c, k, name, device="cpu", dtype=torch.float):
        super().__init__(c=c, k=k, req_val=False, fit_pairwise=True, combine_probs=False, device=device, dtype=dtype, name=name)
        
    def train(self, X, y, val_X=None, val_y=None, verbose=0):
        """Trains lda model for a pair of classes and outputs its coefficients.

        Args:
            X (torch.tensor): Tensor of training predictors. Shape n × c, where n is number of training samples and c is number of combined classifiers.
            y (torch.tensor): Tensor of training labels. Shape n - number of training samples.
            val_X (torch.tensor, optional): Tensor of validation predictors. Shape n × c, 
            where n is number of validation samples and c is number of combined classifiers. Defaults to None.
            val_y ([torch.tensor, optional): Tensor of validation labels. SHape n - number of validation samples. Defaults to None.
            wle (WeightedLinearEnsemble, optional): WeightedLinearEnsemble for which the coefficients are trained. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            torch.tensor: Tensor of model coefficients. Shape 1 × (c + 1), where c is number of combined classifiers.
        """
        clf = LinearDiscriminantAnalysis(solver='lsqr')
        clf.fit(X.cpu(), y.cpu())
        coefs = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                           torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))
        
        return coefs


class logreg(GeneralCombiner):
    """Combining method which uses Logistic Regression to infer combining coefficients.
    """
    def __init__(self, c, k, fit_interc, sweep_C, name, req_val, device="cpu", dtype=torch.float):
        super().__init__(c=c, k=k, req_val=req_val, fit_pairwise=True, combine_probs=False, device=device, dtype=dtype, name=name)
        self.fit_interc_ = fit_interc
        self.sweep_C_ = sweep_C
        
    def train(self, X, y, val_X=None, val_y=None, verbose=0):
        """Trains logistic regression model for a pair of classes and outputs its coefficients.

        Args:
            X (torch.tensor): Tensor of training predictors. Shape n × c, where n is number of training samples and c is number of combined classifiers.
            y (torch.tensor): Tensor of training labels. Shape n - number of training samples.
            val_X (torch.tensor, optional): Tensor of validation predictors. Shape n × c, 
            where n is number of validation samples and c is number of combined classifiers. Defaults to None.
            val_y ([torch.tensor, optional): Tensor of validation labels. SHape n - number of validation samples. Defaults to None.
            wle (WeightedLinearEnsemble, optional): WeightedLinearEnsemble for which the coefficients are trained. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            torch.tensor: Tensor of model coefficients. Shape 1 × (c + 1), where c is number of combined classifiers.
        """
        if self.sweep_C_:
            coefs = _logreg_sweep_C(X, y, val_X=val_X, val_y=val_y, fit_intercept=self.fit_interc_, verbose=verbose, device=self.dev_, dtype=self.dtp_)
        else:
            clf = LogisticRegression(fit_intercept=self.fit_interc_)
            clf.fit(X.cpu(), y.cpu())
            coefs = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                            torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))
            
        return coefs


class average(GeneralCombiner):
    """Combining method which averages logits of combined classifiers.
    """
    def __init__(self, c, k, name, calibrate, combine_probs, req_val, device="cpu", dtype=torch.float):
        super().__init__(c=c, k=k, req_val=req_val, fit_pairwise=False, combine_probs=combine_probs, device=device, dtype=dtype, name=name)
        self.calibrate_ = calibrate
        
    def train(self, X, y, val_X=None, val_y=None, wle=None, verbose=0):
        """Computes and outputs coefficients for averaging model.
        Args:
            X (tensor): Training predictors. Tensor of shape c × n × k, where c is number of combined classifiers, n is numbe of training samples and k is numbe rof classes.
            y (tensor): Training labels. Tensor of shape n - number of training samples.
            val_X (None, optional): Validation predictors. Tensor of shape c × n × k, where c is number of combined classifiers, 
            n is numbe of training samples and k is numbe rof classes. Defaults to None.
            val_y (None, optional): Validation labels. Tensor of shape n - number of validation samples. Defaults to None.
            wle (None, optional): Not used. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
        
        Returns:
            torch.tensor: Computed coefficients. Tensor of shape k × k × (c + 1), where k is number of classes and c in number of combined classifiers.
        """
        coefs = _averaging_coefs(X=X, y=y, val_X=val_X, val_y=val_y, calibrate=self.calibrate_, comb_probs=self.combine_probs_, device=self.dev_, dtype=self.dtp_, verbose=verbose)
        return coefs        


class grad(GeneralCombiner):
    """Combining method which trains its coefficient in an end-to-end manner using gradient descent.
    """
    def __init__(self, c, k, coupling_method, name, device="cpu", dtype=torch.float):
        super().__init__(c=c, k=k, req_val=True, fit_pairwise=False, combine_probs=False, device=device, dtype=dtype, name=name)
        self.coupling_m_ = coupling_method
        
    def train(self, X, y, val_X, val_y, verbose=0):
        """Computes and outputs coefficients.

        Args:
            X (tensor): Training predictors. Tensor of shape c × n × k, where c is number of combined classifiers, n is numbe of training samples and k is numbe rof classes.
            y (tensor): Training labels. Tensor of shape n - number of training samples.
            val_X (None, optional): Validation predictors. Tensor of shape c × n × k, where c is number of combined classifiers, 
            n is numbe of training samples and k is numbe rof classes. Defaults to None.
            val_y (None, optional): Validation labels. Tensor of shape n - number of validation samples. Defaults to None.
            wle (None, optional): Not used. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
        
        Returns:
            torch.tensor: Computed coefficients. Tensor of shape k × k × (c + 1), where k is number of classes and c in number of combined classifiers.
        """
        return _grad_comb(X=val_X, y=val_y, combiner=self, coupling_method=self.coupling_m_, verbose=verbose)

       
comb_methods = {"lda": [lda, {}],
                "logreg": [logreg, {"fit_interc": True, "sweep_C": False, "req_val": False}],
                "logreg_no_interc": [logreg, {"fit_interc": False, "sweep_C": False, "req_val": False}],
                "logreg_sweep_C": [logreg, {"fit_interc": True, "sweep_C": True, "req_val": True}],
                "logreg_no_interc_sweep_C": [logreg, {"fit_interc": False, "sweep_C": True, "req_val": True}],
                "average": [average, {"calibrate": False, "combine_probs": False, "req_val": False}],
                "cal_average": [average, {"calibrate": True, "combine_probs": False, "req_val": True}],
                "prob_average": [average, {"calibrate": False, "combine_probs": True, "req_val": False}],
                "cal_prob_average": [average, {"calibrate": True, "combine_probs": True, "req_val": True}],
                "grad_m1": [grad, {"coupling_method": "m1"}],
                "grad_m2": [grad, {"coupling_method": "m2"}],
                "grad_bc": [grad, {"coupling_method": "bc"}]}


def comb_picker(co_m, c, k, device="cpu", dtype=torch.float):
    if co_m not in comb_methods:
        return None 
    
    return comb_methods[co_m][0](c=c, k=k, device=device, dtype=dtype, name=co_m, **comb_methods[co_m][1])
