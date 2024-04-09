import torch

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes


class Optimizer(Capsule):
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 tag: str = "opt",
                 accelerator: Accelerator = None, 
                 priority: int = 1000) -> None:
        super().__init__(accelerator=accelerator,
                         statefull=False, # this is just a wrapper, no state
                         priority=priority)
        self._optimizer = optimizer
        self._tag = tag
        self._iter_idx = 0
    
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
                err += "same optimizer has been registered twice. "
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

        if self._accelerator.sync_gradients:
            log = {
                f"{self._tag}.lr.{idx}": group.get("lr") 
                for idx, group in enumerate(self._optimizer.param_groups)
            }
            # send log into trackers and reset
            self._accelerator.log(log, step=self._iter_idx)

            if attrs.looper is not None:
                attrs.looper.state.lr = list(log.values())
        
            self._iter_idx += 1


    def destroy(self, attrs: Attributes = None):
        # safe pop from accelerator
        _id = None
        for id, optimizer in enumerate(self._accelerator._optimizers):
            # skip other optimizers if exit
            if optimizer is not self._optimizer:
                continue
            _id = id
            break
        # pop it from list
        if _id is not None:
            self._accelerator._optimizers.pop(_id)
        
        Capsule.destroy(self, attrs=attrs)

    def state_dict(self):
        return Attributes(iter_idx=self._iter_idx)
    
    def load_state_dict(self, state):
        self._iter_idx = state.iter_idx