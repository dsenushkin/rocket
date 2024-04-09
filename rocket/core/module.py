import torch
from typing import List
from operator import attrgetter

from accelerate import Accelerator

from rocket.core.dispatcher import Dispatcher
from rocket.core.capsule import Capsule, Attributes
from rocket.utils import default_move



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
        # if module has already been registered this flag is True
        registered = False
        # loop over all registered models
        for model in self._accelerator._models:
            # skip other models if exist
            if self._module is not model:
                continue
            # if same object found twice, raise exeption
            if registered:
                err = f"{self.__class__.__name__}: "
                err += "same module has been registered twice. "
                raise RuntimeError(err)
            # everything is of, get modified module
            registered = True
            self._module = model
        # module not found, register it
        if not registered:
            # push it in _models and modify forward call
            self._module = self._accelerator.prepare(self._module)
            # safe device placement, necessarily if accelerator 
            # device placement flag is False
            self._module = default_move(self._module, self._accelerator.device)
            #self._module.to(self._accelerator.device)
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
        _id = None
        for id, model in enumerate(self._accelerator._models):
            # skip other modules if exit
            if model is not self._module:
                continue
            _id = id
            break
        # pop it from list
        if _id is not None:
            self._accelerator._models.pop(_id)
        # destroy others
        Dispatcher.destroy(self, attrs=attrs)