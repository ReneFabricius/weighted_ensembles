import re
import torch
from timeit import default_timer as timer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.special import expit
import numpy as np
from abc import abstractmethod

from weensembles.CalibratingMethods import cal_picker
from weensembles.CouplingMethods import coup_picker
from weensembles.predictions_evaluation import compute_acc_topk, compute_nll, compute_pairwise_accuracies
from weensembles.utils import cuda_mem_try, pairwise_accuracies, pairwise_accuracies_penultimate, arguments_dict
from weensembles.PostprocessingMethod import PostprocessingMethod


class GeneralCombiner(PostprocessingMethod):
    """ Abstract class for combining methods.
    """
    def __init__(self, c, k, req_val, uncert, device, dtype, name):
        super().__init__(req_val=req_val or uncert, name=name)
        self.uncert_ = uncert
        self.dev_ = device
        self.dtp_ = dtype
        self.c_ = c
        self.k_ = k
        
    @abstractmethod
    def to_cpu(self):
        """ Moves parameters of the combining method to the cpu memory.
        """
        pass
    
    @abstractmethod
    def to_dev(self):
        """Moves parameters of the combining method to the memory of device specified at the initialization.
        """
        pass
    
    @abstractmethod
    def set_dev(self, device):
        """Sets device for the combining method.
        """
    
    @abstractmethod
    def fit(self, X, y, val_X=None, val_y=None, verbose=0, **kwargs):
        """Trains the combining method.

        Args:
            X (torch.tensor): Predictors.
            y (torch.tensor): Correct labels.
            val_X (torch.tensor, optional): Validation predictors. Defaults to None.
            val_y (torch.tensor, optional): Validation correct labels. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        pass

    @abstractmethod
    def predict_proba(self, X, coupling_method, l=None, verbose=0, batch_size=None, predict_uncertainty=False):
        """Predicts probability based on provided predictors.

        Args:
            X (torch.tensor): Predictors.
            coupling_method (str): Coupling method to use.
            l (int, optional): If specified, performs prediction considering only top l most probable classes from each classifier. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
            batch_size (int, optional): Batch size for processing. Doesnt affect the output, only memory consumption. Defaults to None.
            predict_uncertainty(bool, optional): Whether to compute uncertainty measure. Defaults to False.

        """
        pass
    
    @abstractmethod
    def score(self, X, y, coupling_method):
        pass
 

class GeneralLinearCombiner(GeneralCombiner):
    """Abstract class for linear combining methods.
    """
    
    def __init__(self, c, k, req_val=False, fit_pairwise=False, combine_probs=False, uncert=False, device="cpu", dtype=torch.float, name="no_name"):
        super().__init__(c=c, k=k, req_val=req_val, uncert=uncert, device=device, dtype=dtype, name=name)
        self.fit_pairwise_ = fit_pairwise
        self.combine_probs_ = combine_probs
        self.coefs_ = None
        self.cal_models_ = None
        
    def to_cpu(self):
        if self.coefs_ is not None:
            self.coefs_ = self.coefs_.cpu()
        if self.cal_models_ is not None:
           for cal_m in self.cal_models_:
               cal_m.to_cpu()
                
    def to_dev(self):
        if self.coefs_ is not None:
            self.coefs_ = self.coefs_.to(self.dev_)
        if self.cal_models_ is not None:
            for cal_m in self.cal_models_:
                cal_m.to_dev()
                
    def set_dev(self, device):
        self.dev_ = device
        if self.cal_models_ is not None:
            for cal_m in self.cal_models_:
                cal_m.set_dev(device)

    def fit(self, X, y, val_X=None, val_y=None, verbose=0, **kwargs):
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

        if verbose > 0:
            print("Starting fit with comb method {}".format(self.name_))
        start = timer()
        num = self.k_ * (self.k_ - 1) // 2      # Number of pairs of classes
        print_step = num // 100
        
        if self.uncert_:
            self.cal_models_ = []
            for cler_i in range(self.c_):
                cal_m = cal_picker("TemperatureScaling", device=self.dev_, dtype=self.dtp_) 
                cal_m.fit(logit_pred=X[cler_i], tar=y, verbose=verbose)
                self.cal_models_.append(cal_m)

        if self.fit_pairwise_:
            self.coefs_ = torch.zeros(self.k_, self.k_, self.c_ + 1, device=self.dev_, dtype=self.dtp_)
            if hasattr(self, "sweep_C_") and "save_C" in kwargs and kwargs["save_C"]:
                self.best_C_ = torch.ones(self.k_, self.k_, device=self.dev_, dtype=self.dtp_)
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

                    # s x c tensor of logit supports of c networks for class fc against class sc for s samples
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
                        
                    if hasattr(self, "sweep_C_") and self.sweep_C_:
                        pw_coefs, best_C = pw_coefs
                        if "save_C" in kwargs and kwargs["save_C"]:
                            self.best_C_[fc, sc] = best_C
                            self.best_C_[sc, fc] = best_C
                        
                    self.coefs_[fc, sc, :] = pw_coefs
                    self.coefs_[sc, fc, :] = pw_coefs

                    if verbose > 5:
                        pwacc = pairwise_accuracies_penultimate(SS, pw_y)
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
    def train(self, X: torch.tensor, y: torch.tensor, val_X: torch.tensor=None, val_y: torch.tensor=None, verbose: int=0) -> torch.tensor:
        """Method for computing the combiner coefficients. Method is given predictors and is expected to output coefficients.
        In case of combining methods with fit_pairwise_ == True: 
            X contains logit supports of first class (fc) against the second class (sc), that is logit_fc - logit_sc. The shape of X is s x c,
            where s is number of samples belonging to fc or sc and c is number of combined classifiers.
            y contains labels for these samples. 1 for fc and 0 for sc. Shape s.
            The output is expected to be of shape c + 1 and contain coefficients for each of the combined classifiers and an intercept for this single class pair.
        In case of combining methods with fit_pairwise_ == False:
            X contains logit outputs of the combined classifiers. The shape of X is c x n x k, where c is number of combined classifiers,
            n is number of training samples and k is number of classes.
            y contains labers for these samples from 0 up to k - 1. Shape n.
            The output is expected to be of shape k x k x (c + 1), where k is number of classes and c is number of combined classifiers.
            As the coefficients should be symmetrical, only values above the diagonal are considered.

        Args:
            X (torch.tensor): Predictors.
            y (torch.tensor): Labels.
            val_X (torch.tensor, optional): Predictors for validation. Defaults to None.
            val_y (torch.tensor, optional): Labels for validation. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            torch.tensor: Tensor of computed coefficients.
        """
        pass
    
    def predict_proba(self, X, coupling_method, l=None, verbose=0, batch_size=None, coefs=None,
                      combine_probs=None, predict_uncertainty=False):
        """
        Combines outputs of constituent classifiers using only those classes, which are among the top l most probable
        for some constituent classifier. All coefficients are used, both for classes i, j and j, i.
        :param MP: c x n x k tensor of constituent classifiers posteriors
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param l: how many most probable classes for each constituent classifier to consider
        :param coupling_method: coupling method to use
        :param batch_size: if not none, the size of the batches to use
        :param coefs: k x k x c + 1 tensor of coefficients used for prediction instead of model coefficients.
        :param predict_uncertainty(bool, optional): Whether to compute uncertainty measure. Defaults to False.
        Coefficients are taken as is, they are not necesarilly symmetric.
        :return: n x k tensor of combined posteriors
        """
        if verbose > 0:
            print("Starting predict proba with combining method {} and coupling method {}".format(self.name_, coupling_method))
        coup_m = coup_picker(coupling_method)
        if coup_m is None:
            print("Unknown coupling method {} selected".format(coupling_method))
            return 1
 
        if (self.coefs_ is None and coefs is None) or (self.uncert_ and self.cal_models_ is None): 
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
        unc_list = []

        for start_ind in range(0, n, b_size):
            curMP = X[:, start_ind:(start_ind + b_size), :].to(device=self.dev_, dtype=self.dtp_)
            curn = curMP.shape[1]
            
            if self.uncert_:
                cur_cal_probs = []
                for ci in range(c):
                    ccp = self.cal_models_[ci].predict_proba(logit_pred=curMP[ci])
                    cur_cal_probs.append(ccp.unsqueeze(0))
                
                curMPprob = torch.cat(cur_cal_probs, dim=0) 

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
            NPC = torch.sum(M, dim=1)
            # goes over possible numbers of classes in union of top l classes from each constituent classifier
            for pc in range(l, l * c + 1 if l < self.k_ else l + 1):
                # Pick those samples which have pc classes in the union
                samplM = NPC == pc
                # pcn x c x k tensor
                pcMP = curMP[:, samplM]
                # Pick pc-class masks
                # pcn x k tensor
                pcM = M[samplM]
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
                
                triu_inds = tuple(torch.triu_indices(row=pc, col=pc, offset=1, device=self.dev_))
                if coefs is None:
                    coefs_sym = self.coefs_.transpose(0, 1).index_put(indices=triu_inds, values=torch.tensor([0], dtype=self.dtp_, device=self.dev_))
                else:
                    coefs_sym = coefs.transpose(0, 1).index_put(indices=triu_inds, values=torch.tensor([0], dtype=self.dtp_, device=self.dev_))
                    
                coefs_sym = coefs_sym + coefs_sym.transpose(0, 1)
                
                pc_coefs = coefs_sym.unsqueeze(0).expand(pcn, k, k, c + 1)[pc_coef_M, :].reshape(pcn, pc, pc, c + 1)
                
                Ws = pc_coefs[:, :, :, 0:-1]
                Bs = pc_coefs[:, :, :, -1]
                
                pw_w_supports = pw_supports * Ws
                if (combine_probs is None and self.combine_probs_) or combine_probs:
                    pw_probs = expit(pw_w_supports)
                    pcR = torch.mean(pw_probs, dim=-1)
                else:
                    pcR = expit(torch.sum(pw_w_supports, dim=-1) + Bs)
                                 
                if self.uncert_:
                    pcMPprob = curMPprob[:, samplM]
                    picked_probs = pcMPprob[:, pcM].reshape(c, pcn, pc).transpose(0, 1).transpose(1, 2)
                    exp_probs = picked_probs.unsqueeze(2).expand(pcn, pc, pc, c)
                    coup_probs = exp_probs + exp_probs.transpose(1, 2)
                    # pcn x pc x pc tensor
                    coupR = torch.mean(coup_probs, dim=-1)
                    # coupR contains average of calibrated probabilities that a sample belongs to one of the classes given by indices of the last two dimensions    
                    uncR = torch.full_like(input=pcR, fill_value=0.5)
                    # Add uncertainty to pairwise probabilities according to coupR
                    pcR = coupR * pcR + (1.0 - coupR) * uncR
                    
                pc_probs, pc_uncs = coup_m(pcR, verbose=verbose, out_unc=predict_uncertainty)
                pc_ps = torch.zeros(pcn, k, device=self.dev_, dtype=self.dtp_)
                pc_ps[pcM] = torch.flatten(pc_probs)
                ps[NPC == pc] = pc_ps
                unc_list.append(pc_uncs)
            
            ps_list.append(ps)

        ps_full = torch.cat(ps_list, dim=0)
        if predict_uncertainty:
            unc_full = torch.cat(unc_list, dim=0)
        else:
            unc_full = None
        end = timer()
        if verbose > 0:
            print("Predict proba simple finished in " + str(end - start) + " s")
        
        if predict_uncertainty:
            return ps_full, unc_full
        
        return ps_full

    def score(self, X, y, coupling_method, coefs=None):
        """Computes accuracy of model with given coefficients on given data.

        Args:
            X (torch.tensor): Tensor of testing predictions. Shape n × c. Where c is number of combined classifiers and n is number of samples.
            y (torch.tensor): Tensor of correct classes. Shape n - number of samples.
            coupling_method (string): Coupling method to use for evaluation.
            coefs (torch.tensor): Tensor of model coefficients. Shape 1 × (c + 1). Where c is number of combined classifiers.

        Returns:
            float: Top 1 accuracy of model.
        """
        prob, unc = self.predict_proba(X=X, coupling_method=coupling_method, coefs=coefs if coefs is not None else self.coefs_)
        preds = torch.argmax(prob, dim=1)
        return torch.sum(preds == y).item() / len(y)
    
    def _transform_for_pairwise_fit(self, X, y, batch_size=None):
        """
        Transforms data into format for feeding into pairwise models. Each class must have the same number of samples.
        Resulting features are organized in a class by class manner, this is represented by dimensions k x k.
        Data at position k1, k2 belong either to the class k1 or k2. This can be determined from the tensor of transformed labels where data which belong
        to the class k1 have label 1 and data which belong to the class k2 have label 0.

        Args:
            X (torch.tensor): Input features. Shape c x n x k
            y (torch.tensor): Input labels. Shape n.
            batch_size (int): Batch size for batched processing. Maximal value is number of samples per class and minimal value is 1.

        Returns:
            torch.tensor, torch.tensor, torch.tensor: Returns three tensors.
            First is tensor of transformed features. This has shape 2*nk x k x k x c, where nk is number of samples per class.
            Second is tensor of thansformed labels. This has shape 2*nk x k x k.
            Third is tensor containing upper triangle mask uf the shape k x k.
        """
        c, n, k = X.shape
        nk = n // k
        dev = X.device
        dtp = X.dtype
        
        if batch_size is None:
            batch_size = n // k
        
        # Create boolean mask with upper triangle indices True
        tinds = torch.triu_indices(row=k, col=k, offset=1)
        uppr_mask = torch.zeros(k, k, dtype=torch.bool, device=dev)
        uppr_mask.index_put_(indices=(tinds[0], tinds[1]), values=torch.tensor([True], dtype=torch.bool, device=dev))
        non_diag = torch.eye(k, k, device=dev) != 1
        X_pws = []
        y_pws = []

        labels = torch.arange(k, device=dev, dtype=y.dtype).unsqueeze(1)
        class_indices = torch.nonzero(labels == y, as_tuple=True)[1].reshape(k, -1)
        for bs in range(0, class_indices.shape[1], batch_size):
            batch_inds = class_indices[:, bs:(bs + batch_size)]
            batch_nk = batch_inds.shape[1] 
            pw_inds = batch_inds.unsqueeze(1).expand(k, k, batch_nk) 
            pw_inds = torch.cat([pw_inds, pw_inds.transpose(0, 1)], dim=2)
            row_class = torch.arange(start=0, end=k, device=dev, dtype=torch.long).reshape(k, 1, 1).expand(k, k, 2 * batch_nk)
            col_class = row_class.transpose(0, 1)
            X_pw = X[:, pw_inds, row_class] - X[:, pw_inds, col_class]
            X_pw = torch.permute(X_pw, (3, 1, 2, 0))
            y_pw = torch.cat([torch.ones((batch_nk, k, k), device=dev), torch.zeros((batch_nk, k, k), device=dev)], dim=0)
            
            X_pws.append(X_pw)
            y_pws.append(y_pw)
        
        X_pw = torch.cat(X_pws, dim=0)
        y_pw = torch.cat(y_pws, dim=0)
        
        return X_pw, y_pw.to(dtype=dtp), uppr_mask


class Lda(GeneralLinearCombiner):
    """Combining method which uses Linear DIscriminant Analysis to infer combining coefficients.
    """
    def __init__(self, c, k, uncert, name, req_val, device="cpu", dtype=torch.float):
        super().__init__(c=c, k=k, uncert=uncert, req_val=req_val, fit_pairwise=True, combine_probs=False, device=device, dtype=dtype, name=name)
        
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


class GeneralLogreg(GeneralLinearCombiner):
    def __init__(self, c, k, fit_interc, name, req_val, uncert, fit_pairwise, device="cpu", dtype=torch.float, base_C=1.0):
        super().__init__(c=c, k=k, uncert=uncert, req_val=req_val, fit_pairwise=fit_pairwise,
                       combine_probs=False, device=device, dtype=dtype, name=name)
        self.fit_interc_ = fit_interc
        self.base_C_ = base_C

    def _bce_loss_pw(self, X, y):
        c, n, k = X.shape
        X_pw, y_pw, uppr_mask = self._transform_for_pairwise_fit(X, y)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        Ws = self.coefs_[:, :, 0:-1]
        Bs = self.coefs_[:, :, -1]
        lin_comb = torch.sum(Ws * X_pw, dim=-1) + Bs
            
        loss = bce_loss(torch.permute(lin_comb, (1, 2, 0))[uppr_mask], torch.permute(y_pw, (1, 2, 0))[uppr_mask])
        loss = torch.sum(loss, dim=-1)
        loss /= X_pw.shape[0]
        
        l2_pen = torch.sum(torch.pow(self.coefs_[:, :, :-1][uppr_mask], 2), dim=-1)
        
        loss += l2_pen / (self.base_C_ * c)
        
        tinds = torch.triu_indices(row=k, col=k, offset=1)
        pw_loss = torch.zeros(k, k, dtype=self.dtp_, device=self.dev_)
        pw_loss.index_put_(indices=(tinds[0], tinds[1]), values=loss)

        return pw_loss


class Logreg(GeneralLogreg):
    """Combining method which uses Logistic Regression to infer combining coefficients.
    """
    def __init__(self, c, k, fit_interc, sweep_C, name, req_val, uncert, device="cpu", dtype=torch.float, base_C=1.0):
        super().__init__(c=c, k=k, fit_interc=fit_interc, uncert=uncert, req_val=req_val, fit_pairwise=True, device=device, dtype=dtype, base_C=base_C, name=name)
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
            torch.tensor: Tensor of model coefficients. Shape (c + 1), where c is number of combined classifiers.
        """
        if self.sweep_C_:
            coefs, best_C = self._logreg_sweep_C(X, y, val_X=val_X, val_y=val_y, verbose=verbose)
        else:
            n, c = X.shape
            corrected_C = c * self.base_C_ / (2 * n)
            clf = LogisticRegression(fit_intercept=self.fit_interc_, C=corrected_C)
            clf.fit(X.cpu(), y.cpu())
            coefs = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                            torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))
        if self.sweep_C_:
            return coefs, best_C
        
        return coefs
    
    @torch.no_grad()
    def _logreg_sweep_C(self, X, y, val_X, val_y, verbose=0):
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
        E_start = -3
        E_end = 3
        E_count = 31
        C_vals = 10**np.linspace(start=E_start, stop=E_end,
                            num=E_count, endpoint=True)
        
        best_C = 1.0
        best_acc = 0.0
        best_model = None
        n, c = val_X.shape
        for C_val in C_vals:
            if verbose > 1:
                print("Testing C value {}".format(C_val))
                
            corrected_C_val = C_val * c / (2 * n)
            clf = LogisticRegression(penalty='l2', fit_intercept=self.fit_interc_, verbose=max(verbose - 1, 0), C=corrected_C_val)
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
            
        coefs = torch.cat((torch.tensor(best_model.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                        torch.tensor(best_model.intercept_, device=self.dev_, dtype=self.dtp_)))

        return coefs, best_C


class LogregTorch(GeneralLogreg):
    """Combining method which uses logistic regression implemented in pytorch to infer combining coefficients.
    """
    def __init__(self, c, k, fit_interc, name, req_val, uncert, device="cpu", dtype=torch.float, base_C=1.0, max_iter=15000, tolg=1e-5, tolch=1e-9,
                 learning_rate=0.1, line_search=False, penalty="l2"):
        super().__init__(c=c, k=k, uncert=uncert, req_val=req_val, fit_pairwise=False, fit_interc=fit_interc,
                         base_C=base_C, device=device, dtype=dtype, name=name)
        self.max_iter_ = int(max_iter)
        self.tolg_ = tolg
        self.tolch_ = tolch
        self.learning_rate_ = learning_rate
        if line_search:
            self.line_search_ = 'strong_wolfe'
        else:
            self.line_search_ = None
        assert penalty in ["l1", "l2"]
        self.penalty_ = penalty
        
    def train(self, X, y, val_X=None, val_y=None, verbose=0):
        """Trains logistic regression model for a pair of classes and outputs its coefficients.

        Args:
            X (torch.tensor): Tensor of training predictors. Shape c × n × k, where n is number of training samples,
            c is the number of combined classifiers and k is the number of classes.
            y (torch.tensor): Tensor of training labels. Shape n - number of training samples.
            val_X (torch.tensor, optional): Tensor of validation predictors. Shape n × c, 
            where n is number of validation samples and c is number of combined classifiers. Defaults to None.
            val_y ([torch.tensor, optional): Tensor of validation labels. SHape n - number of validation samples. Defaults to None.
            wle (WeightedLinearEnsemble, optional): WeightedLinearEnsemble for which the coefficients are trained. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            torch.tensor: Tensor of model coefficients. Shape k x k × (c + 1), where c is number of combined classifiers.
        """
        c, n, k = X.shape
        # Expects equal number of samples for each class
        per_class = n // k
        try:
            X_pw, y_pw, upper_mask = cuda_mem_try(fun=lambda batch_size: self._transform_for_pairwise_fit(X, y, batch_size=batch_size),
                                                  start_bsz=per_class, device=self.dev_, dec_coef=0.8, verbose=verbose)
        except RuntimeError as rerr:
            if str(rerr) != "Unsuccessful to perform the requested action. CUDA out of memory." or X.device == "cpu":
                raise rerr
            X_pw, y_pw, upper_mask = cuda_mem_try(fun=lambda batch_size: self._transform_for_pairwise_fit(X.cpu(), y.cpu(), batch_size=batch_size),
                                                  start_bsz=per_class, device=self.dev_, dec_coef=0.8, verbose=verbose)
            X_pw = X_pw.to(device=self.dev_)
            y_pw = y_pw.to(device=self.dev_)
            upper_mask = upper_mask.to(device=self.dev_)

        return cuda_mem_try(fun=lambda batch_size: self._logreg_torch(X_pw=X_pw, y_pw=y_pw, upper_mask=upper_mask,
                                                                      c=c, n=n, k=k, verbose=verbose, micro_batch=batch_size),
                            start_bsz=2 * per_class, device=self.dev_, dec_coef=0.8, verbose=verbose)

    def _logreg_torch(self, X_pw, y_pw, upper_mask, c, n, k, verbose=0, micro_batch=None):
        """Trains multiple logistic regression models using parallelism 

        Args:
            X_pw (torch.tensor): Tensor of training predictors. Shape 2*nk × k × k × c Where c is number of combined classifiers, 
            nk is number of training samples per class and k is number of classes.
            y_pw (torch.tensor): Tensor of training labels. Shape 2*nk × k × k, where nk is number of training samples per class and k is number of classes.
            fit_intercept (bool, optional): Whether to fit intercept. Defaults to True.
            verbose (int, optional): Verbosity level. Defaults to 0.
            max_iter (int, optional): Maximum number of iterations of the LBFGS optimizer. Defaults to 1000.
            micro_batch (_type_, optional): Micro batch size for gradinet accumulation. If None, no micro batching is performed. Defaults to None.

        Returns:
            torch.tensor: fitted coefficients. shape: k x k x c + int(fit_intercept). Only models where k1 < k2 have nonzero coefficients.
        """
        grad_accumulating = micro_batch is not None and micro_batch < (n // k * 2)
        if self.penalty_ == "l1":
            penalty_fun = lambda arg: torch.sum(torch.abs(arg))
        elif self.penalty_ == "l2":
            penalty_fun = lambda arg: torch.sum(torch.pow(arg, 2))
        
        coefs = torch.zeros(size=(k, k, c + int(self.fit_interc_)), device=self.dev_, dtype=self.dtp_, requires_grad=True)
        X_pw.requires_grad_(False)
        y_pw.requires_grad_(False)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
        opt = torch.optim.LBFGS(params=(coefs,), max_iter=self.max_iter_, tolerance_grad=self.tolg_,
                                tolerance_change=self.tolch_, line_search_fn=self.line_search_,
                                lr=self.learning_rate_)
        
        if micro_batch is None:
            micro_batch = X_pw.shape[0]
                        
        def closure_loss():
            opt.zero_grad()
            if self.fit_interc_:
                Ws = coefs[:, :, 0:-1]
                Bs = coefs[:, :, -1]
            
            loss_accum = torch.tensor([0], device=X_pw.device, dtype=X_pw.dtype) 
            for mbs in range(0, X_pw.shape[0], micro_batch):
                cur_X = X_pw[mbs : mbs + micro_batch]
                cur_y = y_pw[mbs : mbs + micro_batch]
                
                if self.fit_interc_:        
                    lin_comb = torch.sum(Ws * cur_X, dim=-1) + Bs
                else:
                    lin_comb = torch.sum(coefs * cur_X, dim=-1)
                
                loss = bce_loss(torch.permute(lin_comb, (1, 2, 0))[upper_mask], torch.permute(cur_y, (1, 2, 0))[upper_mask])
                loss /= X_pw.shape[0]
                
                loss.backward(retain_graph=False)
                loss_accum += loss

            if self.fit_interc_:
                penalty = penalty_fun(coefs[:,:,:-1][upper_mask])
            else:
                penalty = penalty_fun(coefs[upper_mask])
            penalty /= (self.base_C_ * c)
            penalty.backward(retain_graph=False)
            loss_accum += penalty

            return loss_accum
            
        opt.step(closure_loss)
        
        coefs.requires_grad_(False)
        if not self.fit_interc_:
            zero_interc = torch.zeros(k, k, 1, dtype=self.dtp_, device=self.dev_)
            coefs = torch.cat([coefs, zero_interc], dim=-1)
        
        return coefs


class Average(GeneralLinearCombiner):
    """Combining method which averages logits of combined classifiers.
    """
    def __init__(self, c, k, name, calibrate, combine_probs, req_val, uncert, device="cpu", dtype=torch.float):
        super().__init__(c=c, k=k, uncert=uncert, req_val=req_val, fit_pairwise=False, combine_probs=combine_probs, device=device, dtype=dtype, name=name)
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
        coefs = self._averaging_coefs(X=X, y=y, val_X=val_X, val_y=val_y, verbose=verbose)
        return coefs        

    def _averaging_coefs(self, X, y, val_X=None, val_y=None, verbose=0):
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
        if self.calibrate_:
            c, n, k = X.shape
            coefs = torch.zeros(size=(c + 1, ), device=self.dev_, dtype=self.dtp_)
            for ci in range(c):
                ts = cal_picker("TemperatureScaling", device=self.dev_, dtype=self.dtp_)
                ts.fit(X[ci], y, verbose=verbose)
                coefs[ci] = 1.0 / ts.temp_.item()

        else:
            if self.combine_probs_:
                coefs = torch.ones(size=(self.c_ + 1, ), device=self.dev_, dtype=self.dtp_)
            else:
                coefs = torch.full(size=(self.c_ + 1, ), fill_value=1.0 / self.c_, device=self.dev_, dtype=self.dtp_)
            
            coefs[self.c_] = 0
        
        return coefs.expand(self.k_, self.k_, -1)


class Grad(GeneralLinearCombiner):
    """Combining method which trains its coefficient in an end-to-end manner using gradient descent.
    """
    def __init__(self, c, k, coupling_method, uncert, name, device="cpu", dtype=torch.float, base_C=1.0, fit_interc=True):
        super().__init__(c=c, k=k, uncert=uncert, req_val=False, fit_pairwise=False, combine_probs=False, device=device, dtype=dtype, name=name)
        self.coupling_m_ = coupling_method
        self.base_C_ = base_C
        self.fit_interc_ = fit_interc
        
    def train(self, X, y, val_X=None, val_y=None, verbose=0):
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
        return self._grad_comb(X=X, y=y, verbose=verbose)

    def _grad_comb(self, X, y, verbose=0, epochs=10, lr=0.3, momentum=0.85, test_period=None, batch_sz=500):
        """Trains combining coefficients in end-to-end manner by gradient descent method.

        Args:
            X (torch.tensor): Tensor of training predictors. Shape c × n × k. Where c is number of combined classifiers, n is number of training samples and k is number of classes.
            y (torch.tensor): Tensor of training labels. Shape n - number of training samples.
            wle (WeightedLinearEnsemble): WeightedLinearEnsemble model for which the coefficients are trained. Prediction method of this model is needed.
            coupling_method (str): Name of coupling method to be used for training.
            verbose (int, optional): Level of verbosity. Defaults to 0.
            epochs (int, optional): Number of epochs. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 0.3.
            momentum (float, optional): Momentum. Defaults to 0.85.
            test_period (int, optional): If not None, period in which testing pass is performed. Defaults to None.
            batch_sz (int, optional): Batch size. Defaults to 500.

        Raises:
            rerr: Possible cuda memory error.

        Returns:
            torch.tensor: Tensor of model coefficients. Shape k × k × (c + 1). Where k is number of classes and c is number of combined classifiers.
        """
        c, n, k = X.shape
        
        coefs = torch.full(size=(k, k, c + int(self.fit_interc_)), fill_value=1.0 / c, device=self.dev_, dtype=self.dtp_)
        if self.fit_interc_:
            coefs[:, :, c] = 0
        if not self.fit_interc_:
            biases = torch.zeros(size=(k, k, 1), device=self.dev_, dtype=self.dtp_)
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
                test_pred, _ = cuda_mem_try(
                    fun=lambda bsz: self.predict_proba(X=X, l=k, coupling_method=self.coupling_m_, verbose=max(verbose - 2, 0), batch_size=bsz, coefs=coefs if self.fit_interc_ else torch.cat((coefs, biases), dim=-1)),
                    start_bsz=test_bsz, verbose=verbose, device=X.device)
                
                acc = compute_acc_topk(pred=test_pred, tar=y, k=1)
                nll = compute_nll(pred=test_pred, tar=y)
                print("Before training: acc {}, nll {}".format(acc, nll))

        successful_mbatch_size = None
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

                if successful_mbatch_size is None:
                    mbatch_sz = batch_sz
                else:
                    mbatch_sz = successful_mbatch_size
                finished = False
                
                while not finished and mbatch_sz > 0:
                    try:
                        opt.zero_grad()
                        if verbose > 1:
                            print("Trying micro batch size {}".format(mbatch_sz))
                        for mbatch_s in range(0, len(y_batch), mbatch_sz):
                            X_mb = X_batch[:, mbatch_s:(mbatch_s + mbatch_sz)]
                            y_mb = y_batch[mbatch_s:(mbatch_s + mbatch_sz)]
                            pred = thresh(self.predict_proba(X=X_mb, l=k, coupling_method=self.coupling_m_,
                                                                verbose=max(verbose - 2, 0), batch_size=mbatch_sz,
                                                                coefs=coefs if self.fit_interc_ else torch.cat((coefs, biases), dim=-1)))
                            loss = nll_loss(torch.log(pred), y_mb) * (len(y_mb) / len(y_batch))
                            loss.backward()
                        
                        L2 = torch.sum(torch.pow(coefs[:,:,:-1] if self.fit_interc_ else coefs, 2)) / (k * (k - 1) * c) / self.base_C_
                        L2.backward()
                        opt.step()
                        finished = True
                        successful_mbatch_size = mbatch_sz

                    except Exception as rerr:
                        if 'memory' not in str(rerr) and "CUDA" not in str(rerr) and "cuda" not in str(rerr):
                            raise rerr
                        if verbose > 1:
                            print("OOM Exception")
                            print(rerr)
                        del rerr
                        if coefs.grad is not None:
                            del coefs.grad
                        mbatch_sz = int(0.9 * mbatch_sz)
                        with torch.cuda.device(X.device):
                            torch.cuda.empty_cache()
                
            if test_period is not None and (e + 1) % test_period == 0:
                with torch.no_grad():
                    test_pred, _ = cuda_mem_try(
                        fun=lambda bsz: self.predict_proba(X=X, l=k, coupling_method=self.coupling_m_, verbose=max(verbose - 2, 0), batch_size=bsz, coefs=coefs if self.fit_interc_ else torch.cat((coefs, biases), dim=-1)),
                        start_bsz=test_bsz, verbose=verbose, device=X.device)

                    acc = compute_acc_topk(pred=test_pred, tar=y, k=1)
                    nll = compute_nll(pred=test_pred, tar=y)
                    print("Test epoch {}: acc {}, nll {}".format(e, acc, nll))

        coefs.requires_grad_(False)
        return coefs if self.fit_interc_ else torch.cat((coefs, biases), dim=-1)


class Random(GeneralLinearCombiner):
    def __init__(self, c, k, req_val=False, fit_pairwise=False, combine_probs=False, uncert=False, device="cpu", dtype=torch.float, name="random"):
        super().__init__(c=c, k=k, req_val=req_val, fit_pairwise=fit_pairwise, combine_probs=combine_probs, uncert=uncert, device=device, dtype=dtype, name=name)
        
    def train(self, X: torch.tensor = None, y: torch.tensor = None, val_X: torch.tensor = None, val_y: torch.tensor = None, verbose: int = 0) -> torch.tensor:
        # For each pair of classes randomly pick one of the combined classifiers
        rand_pick = torch.randint(low=0, high=self.c_, size=(self.k_, self.k_), device=self.dev_)
        # Transform the pick into one hot encoding
        rand_pick = torch.nn.functional.one_hot(rand_pick)
        # Zeroe values on and under the diagonal.
        triu_inds = tuple(torch.triu_indices(row=self.k_, col=self.k_, offset=0, device=self.dev_))
        rand_pick = rand_pick.transpose(0, 1).index_put(indices=triu_inds, values=torch.tensor([0], device=self.dev_, dtype=rand_pick.dtype)).transpose(0, 1)
        # Add zero intercept
        rand_pick = torch.cat(
            [rand_pick.to(dtype=self.dtp_), torch.zeros(size=(self.k_, self.k_, 1), device=self.dev_, dtype=self.dtp_)],
            dim=2)
        
        return rand_pick


class Net(torch.nn.Module):
    def __init__(self, c, k, device, dtype):
        super().__init__()
        self.c_ = c
        self.k_ = k
        self.dev_ = device
        self.dtp_ = dtype
        
        inputs = c * k
        outputs = k * (k - 1) // 2
        inc = (outputs / inputs) ** (1 / 3)
        fc1_o = int(inputs * inc)
        fc2_o = int(fc1_o * inc)
        self.fc1 = torch.nn.Linear(in_features=inputs, out_features=fc1_o, device=device, dtype=dtype)
        self.fc2 = torch.nn.Linear(in_features=fc1_o, out_features=fc2_o, device=device, dtype=dtype)
        self.fc3 = torch.nn.Linear(in_features=fc2_o, out_features=outputs, device=device, dtype=dtype)
        self.relu = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.special.expit(self.fc3(x))
        
        return x

class Neural(GeneralCombiner):
    """Combining method emplying simple neural network for inferring matrix of pairwise probabilities from classifiers logits.

    """
    def __init__(self, c, k, name, coupling_method, device="cpu", dtype=torch.float, uncert=False):
        """Initializes object.

        Args:
            c (int): Number of combined classifiers.
            k (int): Number of classes.
            name (str): Name of the combining method.
            coupling_method (str): Coupling method to use for training.
            device (str, optional): Device to use. Defaults to "cpu".
            dtype (torch.dtype, optional): Data type to use. Defaults to torch.float.
            uncert (bool, optional): Currently unused. Defaults to False.
        """
        super().__init__(c=c, k=k, req_val=False, uncert=uncert, device=device, dtype=dtype, name=name)
        self.coupling_m_ = coupling_method
        self.net_ = Net(c=c, k=k, device=device, dtype=dtype)
        
    def to_cpu(self):
        """Moves data to cpu.
        """
        self.net_ = self.net_.cpu()

    def to_dev(self):
        """Moves data to device specified at creation of the neural object.
        """
        self.net_ = self.net_.to(device=self.dev_)
    
    def fit(self, X, y, batch_size=500, lr=0.01, momentum=0, epochs=10, verbose=0, test_period=None, val_X=None, val_y=None, **kwargs):
        """Trains neural network used for inferring R matrix using end-to-end training using provided coupling method and validation data.

        Args:
            X (None): Not used.
            y (None): Not used.
            val_X (torch.tensor): Tensor of training predictions. Shape c x n x k, where c is number of combined classifiers, n is number of samples and k is number of classes. 
            val_y (torch.tensor): Tensor of training labels. Shape n - number of samples.
            coupling_method (str): Coupling method to use.
            batch_size (int, optional): Batch size to use for training. Defaults to 500.
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (int, optional): Momentum. Defaults to 0.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        if verbose > 0:
            print("Starting neural fit of {}".format(self.name_)) 
        start = timer()
        
        self.net_.train()
        optimizer = torch.optim.SGD(params=self.net_.parameters(), lr=lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(epochs * 1 / 3), int(epochs * 2 / 3)], gamma=0.1)
        loss_f = torch.nn.NLLLoss()
        thr_value = 1e-9
        thresh = torch.nn.Threshold(threshold=thr_value, value=thr_value)
    
        c, n, k = X.shape
        assert c == self.c_
        assert k == self.k_
        
        for e in range(epochs):
            if verbose > 0:
                print("Processing epoch: {}".format(e))        

            perm = torch.randperm(n=n, device=self.dev_)
            X_perm = X[:, perm]
            y_perm = y[perm]
            for start_ind in range(0, n, batch_size):
                optimizer.zero_grad()
                cur_inp = X_perm[:, start_ind:(start_ind + batch_size)].to(device=self.dev_, dtype=self.dtp_)
                cur_lab = y_perm[start_ind:(start_ind + batch_size)].to(device=self.dev_)
                curn = cur_inp.shape[1]
                pred, _ = cuda_mem_try(
                    fun=lambda bsz: self.predict_proba(X=cur_inp, coupling_method=self.coupling_m_, verbose=verbose - 1, batch_size=bsz),
                    start_bsz=curn,
                    device=self.dev_,
                    verbose=verbose - 1
                )
                loss = loss_f(input=thresh(pred), target=cur_lab)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
                
            if test_period is not None and (e + 1) % test_period == 0:
                self.net_.eval()
                with torch.no_grad():
                    pred, _ = cuda_mem_try(
                        fun=lambda bsz: self.predict_proba(X=X, coupling_method=self.coupling_m_, verbose=verbose - 1, batch_size=bsz),
                        start_bsz=curn,
                        device=self.dev_,
                        verbose=verbose - 1
                    )
                    acc = compute_acc_topk(pred=pred, tar=y, k=1)
                    nll = compute_nll(pred=pred, tar=y)
                print("Epoch: {}, training accuracy: {}, training nll: {}".format(e, acc, nll))
                self.net_.train()
        
        self.net_.eval()
        
        end = timer()
        if verbose > 0:
            print("Fit {} finished in {} s".format(self.name_, end - start))

    def predict_proba(self, X, coupling_method, l=None, verbose=0, batch_size=None):
        """Predicts probability outputs using own neural network and provided coupling method.

        Args:
            X (torch.tensor): Tensor of predictions. Shape c x n x k, where c is number of combined classifiers, n is number of samples and k in number of classes.
            coupling_method (str): Coupling method to use.
            l (int, optional): Currently unused. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
            batch_size (int, optional): Batch size. Defaults to None.

        Returns:
            torch.tensor: Tensor of predicted probabilities. Shape n x k, where n is number of samples and k is number of classes.
        """
        c, n, k = X.shape
        
        assert c == self.c_
        assert k == self.k_

        if verbose > 0:
            print("Starting predict proba of {}".format(self.name_))
        coup_m = coup_picker(coupling_method)
        if coup_m is None:
            print("Unknown coupling method {} selected".format(coupling_method))
            return 1
        
        start = timer()
        
        b_size = batch_size if batch_size is not None else n
        ps_list = []

        for start_ind in range(0, n, b_size):
            curMP = X[:, start_ind:(start_ind + b_size), :].to(device=self.dev_, dtype=self.dtp_)
            curn = curMP.shape[1]
       
            inp = torch.flatten(curMP.transpose(0, 1), start_dim=1)
            trii = torch.tril_indices(row=k, col=k, offset=-1, device=self.dev_)
            out = self.net_(inp)
            R = torch.zeros((curn, k, k), device=self.dev_, dtype=self.dtp_)
            R[:, trii[0], trii[1]] = out
            R = R.transpose(1, 2) + (1.0 - R)
            ps = coup_m(R)
            
            ps_list.append(ps)

        ps_full = torch.cat(ps_list, dim=0)
        end = timer()
        if verbose > 0:
            print("Predict proba neural finished in " + str(end - start) + " s")

        return ps_full

    def score(self, X, y, coupling_method):
        prob, _ = self.predict_proba(X=X, coupling_method=coupling_method)
        preds = torch.argmax(prob, dim=1)
        return torch.sum(preds == y).item() / len(y)
       
       
comb_methods = {"lda": [Lda, {"req_val": False}],
                "logreg": [Logreg, {"fit_interc": True, "sweep_C": False, "req_val": False}],
                "logreg_no_interc": [Logreg, {"fit_interc": False, "sweep_C": False, "req_val": False}],
                "logreg_sweep_C": [Logreg, {"fit_interc": True, "sweep_C": True, "req_val": True}],
                "logreg_no_interc_sweep_C": [Logreg, {"fit_interc": False, "sweep_C": True, "req_val": True}],
                "average": [Average, {"calibrate": False, "combine_probs": False, "req_val": False}],
                "cal_average": [Average, {"calibrate": True, "combine_probs": False, "req_val": False}],
                "prob_average": [Average, {"calibrate": False, "combine_probs": True, "req_val": False}],
                "cal_prob_average": [Average, {"calibrate": True, "combine_probs": True, "req_val": False}],
                "grad_m1": [Grad, {"coupling_method": "m1"}],
                "grad_m2": [Grad, {"coupling_method": "m2"}],
                "grad_bc": [Grad, {"coupling_method": "bc"}],
                "grad_no_interc_m1": [Grad, {"coupling_method": "m1", "fit_interc": False}],
                "grad_no_interc_m2": [Grad, {"coupling_method": "m2", "fit_interc": False}],
                "grad_no_interc_bc": [Grad, {"coupling_method": "bc", "fit_interc": False}],
                "neural_m1": [Neural, {"coupling_method": "m1"}],
                "neural_m2": [Neural, {"coupling_method": "m2"}],
                "neural_bc": [Neural, {"coupling_method": "bc"}],
                "logreg_torch": [LogregTorch, {"fit_interc": True, "req_val": False}],
                "logreg_torch_no_interc": [LogregTorch, {"fit_interc": False, "req_val": False}],
                "logreg_torch_l1": [LogregTorch, {"fit_interc": True, "req_val": False, "penalty": "l1"}],
                "logreg_torch_l1_no_interc": [LogregTorch, {"fit_interc": False, "req_val": False, "penalty": "l1"}],
                "random": [Random, {}]
                }

regularization_coefficients = {
    "logreg": {"base_C": 10 ** (1.2)},
    "logreg_torch": {"base_C": 10 ** (1.2)},
    "logreg.uncert": {"base_C": 10 ** (1.8)},
    "logreg_torch.uncert": {"base_C": 10 ** (1.8)},
    "logreg_no_interc": {"base_C": 10 ** (1.2)},
    "logreg_torch_no_interc": {"base_C": 10 ** (1.2)},
    "logreg_no_interc.uncert": {"base_C": 10 ** (1.6)},
    "logreg_torch_no_interc.uncert": {"base_C": 10 ** (1.6)},
    "grad_bc": {"base_C": 10 ** (-0.4)},
    "grad_no_interc_bc": {"base_C": 10 ** (-0.4)},
    "grad_bc.uncert": {"base_C": 10 ** (0.0)},
    "grad_m1": {"base_C": 10 ** (-0.2)},
    "grad_no_interc_m1": {"base_C": 10 ** (-0.2)},
    "grad_m1.uncert": {"base_C": 10 ** (0.0)},
    "grad_m2": {"base_C": 10 ** (-0.6)},
    "grad_no_interc_m2": {"base_C": 10 ** (-0.6)},
    "grad_m2.uncert": {"base_C": 10 ** (0.0)}   
}


def comb_picker(co_m, c, k, device="cpu", dtype=torch.float):
    """Finds and returns combining method with specified parameters.
    Special parameters can be appended to co_m in a curly braces in a dictionary style.
    Example: logreg_no_interc.uncert{base_C:0.2} sends argument base_C=0.2 to corresponding function from comb_methods dictionary.

    Args:
        co_m (str): Name of combining method. May include specific arguments.
        c (int): Number of classifiers to combine.
        k (int): Number of classes.
        device (str, optional): Device on which to perform the combining method. Defaults to "cpu".
        dtype (str, optional): Datatype of inputs and outputs. Defaults to torch.float.

    Returns:
        GeneralCombiner: Object representing combining method.
    """
    m = re.match(r"^(?P<co>.+?)(\{(?P<args>.*)\})?$", co_m)
    co_m_name = m.group("co")
    args_dict = arguments_dict(m.group("args"))
    co_split = co_m_name.split('.')
    co_name = co_split[0]
    if co_name not in comb_methods:
        return None
    
    if co_m_name in regularization_coefficients:
        reg_coef_name = list(regularization_coefficients[co_m_name].keys())[0]
        if reg_coef_name not in args_dict:
            args_dict[reg_coef_name] = regularization_coefficients[co_m_name][reg_coef_name]
    
    return comb_methods[co_name][0](c=c, k=k, device=device, dtype=dtype, name=co_m, uncert=co_split[-1] == "uncert", **comb_methods[co_name][1], **args_dict)
