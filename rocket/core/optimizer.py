import torch

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes


class Optimizer(Capsule):
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 accelerator: Accelerator = None, 
                 priority: int = 1000) -> None:
        super().__init__(accelerator=accelerator,
                         statefull=False, # this is just a wrapper, no state
                         priority=priority)
        self._optimizer = optimizer
    
    def setup(self, attrs: Attributes=None):
        Capsule.setup(self, attrs=attrs)
        # wrap optimizer for distributed training
        self._accelerator.prepare(self._optimizer)

    def launch(self, attrs: Attributes = None):
        # if training is disabled, nothing to do
        if torch.is_grad_enabled():
            self._optimizer.step()
            self._optimizer.zero_grad()
        # one more log message for humans
        if attrs.looper is not None:
            lrs = [
                group.get("lr") 
                for group in self._optimizer.param_groups
            ]
            attrs.looper.state.lr = lrs

    def destroy(self, attrs: Attributes = None):
        # safe pop from accelerator
        obj = self._accelerator._optimizers.pop()
        # be sure it is the same object
        if obj.optimizer is not self._optimizer:
            err = f"{self.__class__.__name__}: "
            err += "illegal destroy request. "
            err += f"{obj.__class__.__name__}"
            raise RuntimeError(err)
        
        Capsule.destroy(self, attrs=attrs)