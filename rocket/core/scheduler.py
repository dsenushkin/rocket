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
        registered = False
        # loop over all registered schedulers
        for sched in self._accelerator._schedulers:
            # skip other schedulers if exist
            if self._scheduler is not sched.scheduler:
                continue
            # if same object found twice, raise exeption
            if registered:
                err = f"{self.__class__.__name__}: "
                err += "same scheduler has been registered twice. "
                raise RuntimeError(err)
            # everything is ok, get modified optimizer
            registered = True
            self._scheduler = sched
        # optimizer not found, register it
        if not registered:
            # push it in _optimizers and modify forward call
            self._scheduler = self._accelerator.prepare(self._scheduler)

    def launch(self, attrs: Attributes = None):
        # if training is disabled, nothing to do
        if torch.is_grad_enabled():
            self._scheduler.step()
    
    def destroy(self, attrs: Attributes = None):
        # safe pop from accelerator
        _id = None
        for id, scheduler in enumerate(self._accelerator._schedulers):
            # skip other optimizers if exit
            if scheduler is not self._scheduler:
                continue
            _id = id
            break
        # pop it from list
        if _id is not None:
            self._accelerator._schedulers.pop(_id)
        
        Capsule.destroy(self, attrs=attrs)
        