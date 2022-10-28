import abc

import pandas as pd
from timeit import default_timer as timer
from torch.nn import Softmax, CrossEntropyLoss
import torch
import re

from weensembles.predictions_evaluation import ECE_sweep
from weensembles.PostprocessingMethod import PostprocessingMethod
from weensembles.utils import arguments_dict


class CalibratingMethod(PostprocessingMethod):
    def __init__(self, req_val, name, device: str="cpu", dtype: torch.dtype=torch.float32):
        super().__init__(req_val=req_val, name=name)
        self.dev_ = device
        self.dtp_ = dtype
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
        pass


class TemperatureScaling(CalibratingMethod):
    """
    Temperature scaling calibration method.
    """
    def __init__(self, name, start_temp=1.0, max_iter=50, lr=0.01, device=torch.device("cpu"), dtype=torch.float32):
        """
        :param start_temp: Starting temperature. 1.0 means no change.
        :param max_iter: maximum number of iterations of optimizer
        :param lr: learning rate
        """
        super().__init__(req_val=False, name=name, device=device, dtype=dtype)
        self.temp_ = torch.tensor([start_temp], device=device, dtype=dtype)
        self.max_iter_ = max_iter
        self.lr_ = lr

    def _nll_loss(self, temp, logit_pred, tar):
        """
        Computes nll loss for given temperature and data.
        :param temp: Calibrating temperature.
        :param logit_pred: Outputs of penultimate layer. n×k tensor with n samples and k classes.
        :param tar: Correct labels. n tensor with n samples.
        :return: Scalar nll loss.
        """

        cent = CrossEntropyLoss()
        loss = cent(torch.div(logit_pred, temp), tar)

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

        n, k = logit_pred.shape

        if verbose > 1:
            cal_pred = self.predict_proba(logit_pred, self.temp_)
            cur_loss = self._nll_loss(temp=self.temp_, logit_pred=logit_pred, tar=tar).item()
            cur_ece = ECE_sweep(pred=cal_pred, tar=tar)
            print("Strating fit. NLL: {:.4f}, estimated calibration error: {:.4f}".format(cur_loss, cur_ece))
        
        temp = torch.nn.Parameter(self.temp_)
        optimizer = torch.optim.LBFGS([temp], lr=self.lr_, max_iter=self.max_iter_)
        def eval_():
            optimizer.zero_grad()
            loss = self._nll_loss(temp=temp, logit_pred=logit_pred, tar=tar)
            loss.backward()
            return loss
        optimizer.step(eval_)
        
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


cal_methods = {"TemperatureScaling": [TemperatureScaling, {}]}

def cal_picker(cal_m: str, device: str="cpu", dtype: torch.dtype=torch.float32):
    m = re.match(r"^(?P<cal>.+?)(\{(?P<args>.*)\})?$", cal_m)
    cal_m_name = m.group("cal")
    args_dict = arguments_dict(m.group("args"))
    if cal_m_name not in cal_methods:
        return None
    
    return cal_methods[cal_m_name][0](device=device, dtype=dtype, name=cal_m_name, **args_dict)
