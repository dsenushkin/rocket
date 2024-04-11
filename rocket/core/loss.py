import torch

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes



class Loss(Capsule):
    def __init__(self, 
                 objective: torch.nn.Module,
                 tag: str = "train_loss",
                 accelerator: Accelerator = None, 
                 priority: int =1100) -> None:  # higher priority then optimizer
        super().__init__(accelerator=accelerator, 
                         statefull=True,
                         priority=priority)
        self._objective = objective
        self._value = 0.0
        self._tag = tag

    def launch(self, attrs: Attributes = None):
        # if no attributes provided, nothing to do
        if attrs is None:
            return
        # if no batch provided, nothing to do
        if attrs.batch is None:
            return
    
        if not torch.is_grad_enabled():
            return
        
        # main logic here
        loss = self._objective(attrs.batch)

        gathered_loss = self._accelerator.gather(loss).mean()
        self._value += gathered_loss.item() / self._accelerator.gradient_accumulation_steps
        
        # reset loss value if optimizer has steped
        if self._accelerator.sync_gradients:
            # send log into trackers and reset
            if attrs.tracker is not None:
                attrs.tracker.scalars.update({self._tag: self._value})

            if attrs.looper is not None:
                attrs.looper.state.loss = self._value
           
            self._value = 0.0

        self._accelerator.backward(loss)
        

    def state_dict(self):
        return Attributes(value=self._value)
    
    def load_state_dict(self, state):
        self._value = state.value