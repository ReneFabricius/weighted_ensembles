import abc
from abc import ABC

import pandas as pd
from torchmin import minimize
from timeit import default_timer as timer
from torch.nn import Softmax, LogSoftmax, NLLLoss
import torch

from weensembles.predictions_evaluation import ECE_sweep


class CalibrationMethod(ABC):
    """
    Abstract class for calibration method.
    """
    @abc.abstractmethod
    def fit(self, logit_pred, tar):
        """
        Fit calibration method using logits of probabilities or outputs of penultimate layer.
        :param logit_pred: logit outputs
        :param tar: correct class labels
        :return:
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, logit_pred):
        """
        Predicts probability using the fitted method
        :param logit_pred: logit outputs
        :return: calibrated probabilities
        """
        pass

    @abc.abstractmethod
    def get_model_coefs(self):
        """
        :return: Model coefficients in the form of a pandas DataFrame
        """
        pass
    
    @abc.abstractmethod
    def to_cpu():
        """Moves tensor attributes to cpu.
        """
        pass
    
    @abc.abstractmethod
    def to_dev():
        """Moves tensor attributes to self.dev_ 
        """
        pass

    @abc.abstractmethod
    def set_dev(self, device):
        """Sets device for the model.

        Args:
            device (string): device
        """


class TemperatureScaling(CalibrationMethod):
    """
    Temperature scaling calibration method.
    """
    def __init__(self, start_temp=1.0, max_iter=50, device=torch.device("cpu"), dtp=torch.float32):
        """
        :param start_temp: Starting temperature. 1.0 means no change.
        :param max_iter: maximum number of iterations of optimizer
        """
        self.temp_ = torch.tensor([start_temp], device=device, dtype=dtp)
        self.max_iter_ = max_iter
        self.dev_ = device
        self.dtp_ = dtp

    def _nll_loss(self, temp, logit_pred, tar):
        """
        Computes nll loss for given temperature and data.
        :param temp: Calibrating temperature.
        :param logit_pred: Outputs of penultimate layer. n×k tensor with n samples and k classes.
        :param tar: Correct labels. n tensor with n samples.
        :return: Scalar nll loss.
        """

        l_sm = LogSoftmax(dim=1)
        nll = NLLLoss()
        loss = nll(l_sm(torch.div(logit_pred, temp)), tar)

        return loss

    @torch.no_grad()
    def fit(self, logit_pred, tar, verbose=0, solver="BFGS"):
        """
        Fits model to provided data.
        :param solver: Solver to use for optimization.
        :param logit_pred: Outputs of penultimate layer. n×k tensor with n samples and k classes.
        :param tar: Correct labels. n tensor with n samples.
        :param verbose: Print additional info.
        :return:
        """

        start = timer()

        self.temp_.requires_grad_(True)        
        n, k = logit_pred.shape

        if verbose > 1:
            cal_pred = self.predict_proba(logit_pred, self.temp_)
            cur_loss = self._nll_loss(temp=self.temp_, logit_pred=logit_pred, tar=tar).item()
            cur_ece = ECE_sweep(pred=cal_pred, tar=tar)
            print("Strating fit. NLL: {:.4f}, estimated calibration error: {:.4f}".format(cur_loss, cur_ece))

        opt = minimize(
            fun=lambda tmp: self._nll_loss(temp=tmp, logit_pred=logit_pred, tar=tar),
            x0=self.temp_,
            max_iter= self.max_iter_,
            method=solver,
            disp=verbose)
        self.temp_ = opt.x
        
        self.temp_.requires_grad_(False)
        
        end = timer()

        if verbose > 1:
            cal_pred = self.predict_proba(logit_pred, self.temp_)
            cur_loss = self._nll_loss(temp=self.temp_, logit_pred=logit_pred, tar=tar).item()
            cur_ece = ECE_sweep(pred=cal_pred, tar=tar)
            print("Fit finished in {:.4f}s. NLL: {:.4f}, estimated calibration error: {:.4f}".format(end - start, cur_loss, cur_ece))

        return 0

    @torch.no_grad()
    def predict_proba(self, logit_pred, temp=None):
        """
        Computes calibrated probabilities.
        :param temp: Temperature for calibration. Default - model trained temperature.
        :param logit_pred: Outputs of penultimate layer. n×k tensor with n samples and k classes.
        :return: n×k tensor of calibrated probabilities.
        """

        sftm = Softmax(dim=-1)

        if temp is None:
            return sftm(logit_pred / self.temp_.item())

        return sftm(logit_pred / temp.item())

    @torch.no_grad()
    def get_model_coefs(self):
        """
        :return: DataFrame with temperature.
        """
        coefs = {"temperature": self.temp_.item()}
        df = pd.DataFrame(data=coefs, index=[0])

        return df
    
    @torch.no_grad()
    def to_cpu(self):
        self.temp_ = self.temp_.cpu()
        
    @torch.no_grad()
    def to_dev(self):
        self.temp_ = self.temp_.to(self.dev_)
        
    @torch.no_grad()
    def set_dev(self, device):
        self.dev_ = device
