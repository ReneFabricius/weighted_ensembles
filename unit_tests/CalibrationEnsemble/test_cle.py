import unittest
import torch

from weensembles.CalibrationEnsemble import CalibrationEnsemble
from unit_tests.test_utils import load_networks_outputs

class Test_CLEFit(unittest.TestCase):
    net_outputs_folder = "./unit_tests/input_data"
    c = 3
    k = 10
    
    def _TempSc_fit(self, device, dtype):
        cle = CalibrationEnsemble(c=Test_CLEFit.c, k=Test_CLEFit.k, device=device, dtp=dtype)
        net_outputs = load_networks_outputs(nn_outputs_path=Test_CLEFit.net_outputs_folder,
                                            device=device, dtype=dtype,
                                            load_train_data=False)
        
        cle.fit(preds=net_outputs["val_outputs"], labels=net_outputs["val_labels"],
                calibration_method="TemperatureScaling")

        for cal_model in cle.cal_models_:
            assert(not torch.isnan(cal_model.temp_).any())
            assert(cal_model.temp_.dtype == dtype)
            assert(cal_model.temp_.device.type == device)

    def test_cle_fit_TempSc_cpu_float(self):
        self._TempSc_fit(device="cpu", dtype=torch.float32)
        
    def test_cle_fit_TempSc_cuda_float(self):
        self._TempSc_fit(device="cuda", dtype=torch.float32)
        
    def test_cle_fit_TempSc_cuda_double(self):
        self._TempSc_fit(device="cuda", dtype=torch.float64)
        

class Test_CLEPredict(unittest.TestCase):
    net_outputs_folder = "./unit_tests/input_data"
    c = 3
    k = 10
   
    def _TempSc_predict(self, device, dtype):
        cle = CalibrationEnsemble(c=Test_CLEPredict.c, k=Test_CLEPredict.k, device=device, dtp=dtype)
        net_outputs = load_networks_outputs(nn_outputs_path=Test_CLEPredict.net_outputs_folder,
                                            device=device, dtype=dtype,
                                            load_train_data=False)
        
        cle.fit(preds=net_outputs["val_outputs"], labels=net_outputs["val_labels"],
                calibration_method="TemperatureScaling")
        
        prediction = cle.predict_proba(preds=net_outputs["test_outputs"])
        
        tc, tn, tk = net_outputs["test_outputs"].shape
        print(torch.min(prediction))
        print(torch.max(prediction))

        assert(prediction.shape == (tn, tk))
        assert(not torch.isnan(prediction).any())
        assert(not (prediction < 0).any())
        assert(not (prediction > 1).any())
        preds_sum = torch.sum(prediction, dim=1)
        assert(torch.isclose(preds_sum, torch.ones_like(preds_sum)).all())
        assert(prediction.device.type == device)
        assert(prediction.dtype == dtype)
        
        return prediction

    def test_cle_predict_TempSc_cuda_double(self):
        self._TempSc_predict(device="cuda", dtype=torch.float64)