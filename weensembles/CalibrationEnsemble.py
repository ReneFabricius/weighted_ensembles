import torch
import pickle
import pandas as pd

from weensembles.Ensemble import Ensemble
from weensembles.CalibratingMethods import cal_picker

class CalibrationEnsemble(Ensemble):
    def __init__(self, c=0, k=0, device=torch.device("cpu"), dtp=torch.float32):
        """
        Trainable ensembling of classification posteriors.
        :param c: Number of classifiers
        :param k: Number of classes
        :param device: Torch device to use for computations
        :param dtp: Torch datatype to use for computations
        """
        super().__init__(c=c, k=k, device=device, dtp=dtp)
        self.cal_models_ = [None for _ in range(c)]
        self.constituent_names_ = None

    @torch.no_grad()
    def fit(self, preds, labels, calibration_method, verbose=0, constituent_names=None, **kwargs):
        """
        Fit the calibration model for each combined classifier.
        :param preds: Predictions from penultimate layer or logits.
        c×n×k tensor with c classifiers, n samples and k classes
        :param labels: Correct labels. n tensor with n samples.
        :param calibration_method: Calibration method to use.
        :param verbose: Print extra info.
        :param constituent_names: Correctly ordered list of ensemble constituent names.
        :return:
        """

        c, n, k = preds.shape

        assert c == self.c_
        assert k == self.k_
        
        self.constituent_names_ = constituent_names

        for ci in range(c):
            cal_m_obj = cal_picker(calibration_method, device=self.dev_, dtype=self.dtp_)
            if cal_m_obj is None:
                raise ValueError("Unknown calibrating method selected: {}".format(calibration_method))

            self.cal_models_[ci] = cal_m_obj 
            self.cal_models_[ci].fit(preds[ci], labels, verbose=verbose)

    @torch.no_grad()
    def predict_proba(self, preds, output_net_preds=False):
        """
        Combines the outputs of classifiers and produces probabilities.
        :param output_net_preds: If True, method also outputs calibrated network predictions.
        :param preds: Penultimate layer outputs or logits to combine.
        c×n×k tensor with c classifiers, n samples and k classes
        :return: Predicted probabilities. n×k tensor with n samples and k classes
        """
        if None in self.cal_models_:
            print("Error: ensemble was not trained.")
            return 1

        c, n, k = preds.shape

        assert c == self.c_
        assert k == self.k_

        cal_MP_l = []
        for ci in range(c):
            cal_prob = self.cal_models_[ci].predict_proba(preds[ci].cpu())
            cal_MP_l.append(cal_prob.unsqueeze(0))

        cal_MP = torch.cat(cal_MP_l, dim=0).to(dtype=self.dtp_, device=self.dev_)

        prob = torch.sum(input=cal_MP, dim=0)
        prob = prob / c

        if output_net_preds:
            return prob, cal_MP

        return prob

    @torch.no_grad()
    def save(self, file, verbose=0):
        """
        Save ensemble into a file.
        :param file: file to save the ensemble to
        :return:
        """
        if verbose > 0:
            print("Saving models into file: " + str(file))
        
        for cal_m in self.cal_models_:
            if cal_m is not None:
                cal_m.to_cpu()
        
        with open(file, 'wb') as f:
            pickle.dump(self.__dict__, f)
            
        for cal_m in self.cal_models_:
            if cal_m is not None:
                cal_m.to_dev()

    @torch.no_grad()
    def load(self, file, verbose=0):
        """
        Load ensemble from a file
        :param file: file to load the ensemble from
        :return:
        """
        if verbose > 0:
            print("Loading models from file: " + str(file))
        with open(file, 'rb') as f:
            dump_dict = pickle.load(f)
            
        self.__dict__.update(dump_dict)
        
        for cal_m in self.cal_models_:
            if cal_m is not None:
                cal_m.to_dev()

    @torch.no_grad()
    def save_coefs_csv(self, file):
        """
        Save calibration coefficients into a csv file.
        :param file: file to save the coefficients to
        :return:
        """
        dfs = []
        for i in range(self.c_):
            net_df = self.cal_models_[i].get_model_coefs()
            net_df["network"] = "network" + str(i + 1)
            dfs.append(net_df)

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(file, index=False)
