from abc import ABC, abstractmethod

class Ensemble(ABC):
    def __init__(self, c, k, device, dtp):
        self.c_ = c
        self.k_ = k
        self.dev_ = device
        self.dtp_ = dtp

    @abstractmethod
    def fit(self, preds, labels, postprocessing_method,
            verbose=0, val_preds=None, val_labels=None, **kwargs):
        pass
    
    @abstractmethod
    def save(self, file, verbose=0):
        pass
    
    @abstractmethod
    def load(self, file, verbose=0):
        pass
 
    @abstractmethod
    def save_coefs_csv(self, file):
        pass