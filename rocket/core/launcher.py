import logging
from typing import List

from accelerate import Accelerator

from rocket.core.dispatcher import Dispatcher
from rocket.core.capsule import Attributes, Capsule
from rocket.core.loop import Looper


class Launcher(Dispatcher):
    def __init__(self,
                 capsules: List[Capsule],
                 num_epochs: int=1,
                 statefull: bool=False,
                 accelerator: Accelerator=None):
        super().__init__(capsules=capsules,
                         accelerator=accelerator)
        self._num_epochs = num_epochs
        self._epoch_idx = 0
        self._statefull = statefull

    def set(self, attrs: Attributes = None):
        pass

    def reset(self, attrs: Attributes = None):
        pass

    def launch(self, attrs: Attributes=None):
        # default debug log
        Capsule.launch(self, attrs=attrs)

        attrs = attrs or Attributes()
        # setup loops
        self.setup(attrs=attrs)

        for _epoch in range(self._epoch_idx, self._num_epochs):
            # epoch set logic
            # run full loop logic sequentially
            for capsule in self._capsules:
                capsule.set(attrs=attrs)
                # launch logic
                capsule.launch(attrs=attrs)
                # epoch reset logic
                capsule.reset(attrs=attrs)
            self._epoch_idx = _epoch
        # patched dispatcher call
        self.destroy(attrs)


    def destroy(self, attrs: Attributes):
        # call others
        Dispatcher.destroy(self, attrs=attrs)
        # finish training
        self._accelerator.end_training()


    def state_dict(self):
        return Attributes(epoch_idx=self._epoch_idx)
    

    def load_state_dict(self, state):
        self._epoch_idx = state.epoch_idx
