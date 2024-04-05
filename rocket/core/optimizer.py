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
        registered = False
        # loop over all registered optimizers
        for optim in self._accelerator._optimizers:
            # skip other optimizers if exist
            if self._optimizer is not optim.optimizer:
                continue
            # if same object found twice, raise exeption
            if registered:
                err = f"{self.__class__.__name__}: "
                err += "same scheduler has been registered twice. "
                raise RuntimeError(err)
            # everything is ok, get modified optimizer
            registered = True
            self._optimizer = optim
        # optimizer not found, register it
        if not registered:
            # push it in _optimizers and modify forward call
            self._optimizer = self._accelerator.prepare(self._optimizer)


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
        for id, optimizer in enumerate(self._accelerator._optimizers):
            # skip other optimizers if exit
            if optimizer is not self._optimizer:
                continue
            # pop it from list
            self._accelerator._optimizers.pop(id)
        
        Capsule.destroy(self, attrs=attrs)