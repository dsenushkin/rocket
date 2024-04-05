from typing import List

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes



class Metric(Capsule):
    def __init__(self,
                 accelerator: Accelerator=None, 
                 priority: int=1000) -> None:
        super().__init__(accelerator=accelerator, 
                         priority=priority)

    def launch(self, attrs: Attributes = None):
        raise NotImplementedError(f"{self.__class__.__name__}: " + \
                                  "metric should implement launch()")
        
    def reset(self, attrs: Attributes = None):
        raise NotImplementedError(f"{self.__class__.__name__}: " + \
                                  "metric should implement reset()")
        