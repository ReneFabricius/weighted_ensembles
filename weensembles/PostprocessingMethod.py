from abc import ABC

class PostprocessingMethod(ABC):
    def __init__(self, req_val, name):
        self.req_val_ = req_val
        self.name_ = name