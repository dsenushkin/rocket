import torch

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes



class Loss(Capsule):
    def __init__(self, 
                 objective: torch.nn.Module,
                 accelerator: Accelerator = None, 
                 priority: int =1100) -> None:  # higher priority then optimizer
        super().__init__(accelerator=accelerator, 
                         statefull=False, # this is just a wrapper, no state
                         priority=priority)
        self._objective = objective

    def launch(self, attrs: Attributes = None):
        # if no attributes provided, nothing to do
        if attrs is None:
            return
        # if no batch provided, nothing to do
        if attrs.batch is None:
            return
        # main logic here
        loss = self._objective(attrs.batch)
        self._accelerator.backward(loss)

        if attrs.looper is not None:
            attrs.looper.state.loss = loss.item()
        