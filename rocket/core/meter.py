from typing import List

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes
from rocket.core.dispatcher import Dispatcher


class Meter(Dispatcher):
    def __init__(self, 
                 capsules: List[Capsule], 
                 keys: List,
                 accelerator: Accelerator = None, 
                 priority=1000) -> None:
        super().__init__(capsules=capsules, 
                         accelerator=accelerator, 
                         priority=priority)
        self._keys = keys
        
    def launch(self, attrs: Attributes = None):
        # nothing to do if empty data flow passed
        if attrs is None or attrs.batch is None:
            return
        
        # main logic
        try:
            inputs = [attrs.batch[key] for key in self._keys]
            gathered_inputs = self._accelerator.gather_for_metrics(*inputs)
            # store gathered data into batch for further processing
            # be ware of capsule ordering, it may happen that not all of 
            # the original batch keys will be gathered that leads to 
            # different batch sizes in different keys of the batch
            for idx, key in enumerate(self._keys):
                attrs.batch[key] = gathered_inputs[idx]
        except:
            err = f"{self.__class__.__name__}: some keys are not found in batch"
            raise RuntimeError(err)
        
        Dispatcher.launch(self, attrs=attrs)
        

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
        