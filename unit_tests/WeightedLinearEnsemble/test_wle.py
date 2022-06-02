import unittest
import torch

from weensembles.WeightedLinearEnsemble import WeightedLinearEnsemble
from unit_tests.test_utils import load_networks_outputs

class Test_WLEFit(unittest.TestCase):
    net_outputs_folder = "./unit_tests/input_data"
    c = 3
    k = 10
    
    def _wle_fit(self, comb_m, device, dtype):
        wle = WeightedLinearEnsemble(c=Test_WLEFit.c, k=Test_WLEFit.k, device=device, dtp=dtype)
        net_outputs = load_networks_outputs(nn_outputs_path=Test_WLEFit.net_outputs_folder,
                                            device=device, dtype=dtype,
                                            load_train_data=False)
        
        wle.fit(preds=net_outputs["val_outputs"], labels=net_outputs["val_labels"],
                combining_method=comb_m)

        coefficients = wle.comb_model_.coefs_
        assert(coefficients.shape == (Test_WLEFit.k, Test_WLEFit.k, Test_WLEFit.c + 1))
        assert(not torch.isnan(coefficients).any())
        assert(coefficients.dtype == dtype)
        assert(coefficients.device.type == device)
        
    def test_wle_fit_average_cpu_float(self):
        self._wle_fit(comb_m="average", device="cpu", dtype=torch.float32)
        
    def test_wle_fit_cal_average_cpu_float(self):
        self._wle_fit(comb_m="cal_average", device="cpu", dtype=torch.float32)
  
    def test_wle_fit_prob_average_cpu_float(self):
        self._wle_fit(comb_m="prob_average", device="cpu", dtype=torch.float32)

    def test_wle_fit_cal_prob_average_cpu_float(self):
        self._wle_fit(comb_m="cal_prob_average", device="cpu", dtype=torch.float32)

    def test_wle_fit_grad_m1_cpu_float(self):
        self._wle_fit(comb_m="grad_m1", device="cpu", dtype=torch.float32)
    
    def test_wle_fit_grad_m2_cpu_float(self):
        self._wle_fit(comb_m="grad_m2", device="cpu", dtype=torch.float32)

    def test_wle_fit_grad_bc_cpu_float(self):
        self._wle_fit(comb_m="grad_bc", device="cpu", dtype=torch.float32)
        
    def test_wle_fit_logreg_cpu_float(self):
        self._wle_fit(comb_m="logreg", device="cpu", dtype=torch.float32)

    def test_wle_fit_logreg_torch_cpu_float(self):
        self._wle_fit(comb_m="logreg_torch", device="cpu", dtype=torch.float32)

    def test_wle_fit_logreg_no_interc_cpu_float(self):
        self._wle_fit(comb_m="logreg_no_interc", device="cpu", dtype=torch.float32)

    def test_wle_fit_logreg_torch_no_interc_cpu_float(self):
        self._wle_fit(comb_m="logreg_torch_no_interc", device="cpu", dtype=torch.float32)
    
    def test_wle_fit_lda_cpu_float(self):
        self._wle_fit(comb_m="lda", device="cpu", dtype=torch.float32)

    def test_wle_fit_average_cuda_double(self):
        self._wle_fit(comb_m="average", device="cuda", dtype=torch.float64)
        
    def test_wle_fit_cal_average_cuda_double(self):
        self._wle_fit(comb_m="cal_average", device="cuda", dtype=torch.float64)
  
    def test_wle_fit_prob_average_cuda_double(self):
        self._wle_fit(comb_m="prob_average", device="cuda", dtype=torch.float64)

    def test_wle_fit_cal_prob_average_cuda_double(self):
        self._wle_fit(comb_m="cal_prob_average", device="cuda", dtype=torch.float64)

    def test_wle_fit_grad_m1_cuda_double(self):
        self._wle_fit(comb_m="grad_m1", device="cuda", dtype=torch.float64)
    
    def test_wle_fit_grad_m2_cuda_double(self):
        self._wle_fit(comb_m="grad_m2", device="cuda", dtype=torch.float64)

    def test_wle_fit_grad_bc_cuda_double(self):
        self._wle_fit(comb_m="grad_bc", device="cuda", dtype=torch.float64)
        
    def test_wle_fit_logreg_cuda_double(self):
        self._wle_fit(comb_m="logreg", device="cuda", dtype=torch.float64)

    def test_wle_fit_logreg_torch_cuda_double(self):
        self._wle_fit(comb_m="logreg_torch", device="cuda", dtype=torch.float64)

    def test_wle_fit_logreg_no_interc_cuda_double(self):
        self._wle_fit(comb_m="logreg_no_interc", device="cuda", dtype=torch.float64)

    def test_wle_fit_logreg_torch_no_interc_cuda_double(self):
        self._wle_fit(comb_m="logreg_torch_no_interc", device="cuda", dtype=torch.float64)
    
    def test_wle_fit_lda_cuda_double(self):
        self._wle_fit(comb_m="lda", device="cuda", dtype=torch.float64)
    

class Test_WLEPredict(unittest.TestCase):
    net_outputs_folder = "./unit_tests/input_data"
    c = 3
    k = 10

    def _wle_predict(self, coup_m, device, dtype, l, bsz):
        wle = WeightedLinearEnsemble(c=Test_WLEFit.c, k=Test_WLEFit.k, device=device, dtp=dtype)
        net_outputs = load_networks_outputs(nn_outputs_path=Test_WLEFit.net_outputs_folder,
                                            device=device, dtype=dtype,
                                            load_train_data=False)
        
        wle.fit(preds=net_outputs["val_outputs"], labels=net_outputs["val_labels"],
                combining_method="logreg_torch")
        
        prediction = wle.predict_proba(preds=net_outputs["test_outputs"], coupling_method=coup_m,
                                       l=l, batch_size=bsz)
        
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
    '''
    m1 produces negative numbers
    def test_wle_predict_m1_cpu_float(self):
        self._wle_predict(coup_m="m1", device="cpu", dtype=torch.float32, l=Test_WLEPredict.k, bsz=None)
    '''
    
    def test_wle_predict_m2_cpu_float(self):
        self._wle_predict(coup_m="m2", device="cpu", dtype=torch.float32, l=Test_WLEPredict.k, bsz=None)
 
    def test_wle_predict_bc_cpu_float(self):
        self._wle_predict(coup_m="bc", device="cpu", dtype=torch.float32, l=Test_WLEPredict.k, bsz=None)
 
    def test_wle_predict_sbt_cpu_float(self):
        self._wle_predict(coup_m="sbt", device="cpu", dtype=torch.float32, l=Test_WLEPredict.k, bsz=None)
 
    def test_wle_predict_m2_cuda_double(self):
        self._wle_predict(coup_m="m2", device="cuda", dtype=torch.float64, l=Test_WLEPredict.k, bsz=None)
 
    def test_wle_predict_bc_cuda_double(self):
        self._wle_predict(coup_m="bc", device="cuda", dtype=torch.float64, l=Test_WLEPredict.k, bsz=None)
 
    def test_wle_predict_sbt_cuda_double(self):
        self._wle_predict(coup_m="sbt", device="cuda", dtype=torch.float64, l=Test_WLEPredict.k, bsz=None)

    def test_wle_predict_batching_m2_cuda_double(self):
        pred10 = self._wle_predict(coup_m="m2", device="cuda", dtype=torch.float64, l=Test_WLEPredict.k, bsz=10)
        pred100 = self._wle_predict(coup_m="m2", device="cuda", dtype=torch.float64, l=Test_WLEPredict.k, bsz=100)
        assert(torch.isclose(pred10, pred100).all())
