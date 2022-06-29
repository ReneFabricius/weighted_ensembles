from abc import ABC, abstractmethod
from pyrsistent import l
import torch

class OODDetector(ABC):
    def __init__(self, name: str) -> None:
        self.name_ = name
        
    @abstractmethod
    def get_scores(self, logits: torch.tensor) -> torch.tensor:
        pass
    

class MaximumSoftmaxProbability(OODDetector):
    def __init__(self):
        super().__init__("MSP")
        
    def get_scores(self, logits: torch.tensor) -> torch.tensor:
        """Computes ood scores as 1 minus maximum softmax probability.

        Args:
            logits (torch.tensor): Output logits of classifier.

        Returns:
            torch.tensor: 1 - maximum softmax probability for each sample.
        """
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(logits)
        max_probs, _ = torch.max(probs, dim=-1)
        return 1 - max_probs
    
class MaximumLogit(OODDetector):
    def __init__(self) -> None:
        super().__init__(name="ML")
        
    def get_scores(self, logits: torch.tensor) -> torch.tensor:
        """Computes ood score as 1 minus normalized max logits.

        Args:
            logits (torch.tensor): Output logits of classifier.

        Returns:
            torch.tensor: Maximum logit for each sample normalized to the [0, 1] interval.
        """
        max_log, _ = torch.max(logits, dim=-1)
        min_score = torch.min(max_log)
        max_score = torch.max(max_log)
        max_log = (max_log - min_score) / (max_score - min_score)
        return 1 - max_log