import torch

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes


class Scheduler(Capsule):
    def __init__(self, 
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 accelerator: Accelerator = None, 
                 priority: int = 1000) -> None:
        super().__init__(accelerator=accelerator, 
                         statefull=False, # this is just a wrapper, no state
                         priority=priority)
        self._scheduler = scheduler

    def setup(self, attrs: Attributes=None):
        Capsule.setup(self, attrs=attrs)
        # wrap scheduler for distributed training
        self._accelerator.prepare(self._scheduler)

    def launch(self, attrs: Attributes = None):
        # if training is disabled, nothing to do
        if torch.is_grad_enabled():
            self._scheduler.step()
    
    def destroy(self, attrs: Attributes = None):
        # safe pop from accelerator
        obj = self._accelerator._schedulers.pop()
        # be sure it is the same object
        if obj.scheduler is not self._scheduler:
            err = f"{self.__class__.__name__}: "
            err += "illegal destroy request. "
            err += f"{obj.__class__.__name__}"
            raise RuntimeError(err)
        
        Capsule.destroy(self, attrs=attrs)
        