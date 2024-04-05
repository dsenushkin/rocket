import torch
from typing import List
from operator import attrgetter

from accelerate import Accelerator

from rocket.core.dispatcher import Dispatcher
from rocket.core.capsule import Capsule, Attributes



class Module(Dispatcher):
    def __init__(self, 
                 module: torch.nn.Module, 
                 capsules: List[Capsule] = [], # suppose to include 
                                               # losses, optimizers, 
                                               # schedulers, postprocessors
                 accelerator: Accelerator = None,
                 priority: int = 1000) -> None:
        super().__init__(capsules=capsules,
                         accelerator=accelerator, 
                         priority=priority)
        self._module = module

    def setup(self, attrs: Attributes=None):
        # register model BEFORE other entities 
        self.check_accelerator()
        # register distributed training
        self._module = self._accelerator.prepare(self._module)
        # might be unnecessary, accelerate handle device placement automatically
        self._module = self._module.to(self._accelerator.device)
        # call others
        Dispatcher.setup(self, attrs)
    

    def launch(self, attrs: Attributes=None):
        # module expects non-empty attributes dict 
        # and non-empty "batch" key
        if attrs is None or attrs.batch is None: 
            return

        # train/eval handler via grad_enabled flag
        if torch.is_grad_enabled():
            # training mode
            self._module.train()
        else:
            # eval mode
            self._module.eval()

        # if accumulation is enabled, it prevents unnecessary grad sync
        with self._accelerator.accumulate(self._module):
            # forward model
            attrs.batch = self._module.forward(attrs.batch)
            # apply losses, optimizers and schedulers
            Dispatcher.launch(self, attrs=attrs)

    def destroy(self, attrs: Attributes=None):
        # safe pop model from accelerator
        obj = self._accelerator._models.pop()
        # be sure it is the same object
        if obj is not self._module:
            err = f"{self.__class__.__name__}: "
            err += "illegal destroy request. "
            err += f"{obj.__class__.__name__}"
            raise RuntimeError(err)
        # destroy others
        Dispatcher.destroy(self, attrs=attrs)