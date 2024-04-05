import torch
from typing import List
from tqdm import tqdm
from termcolor import colored

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes
from rocket.core.dataset import Dataset
from rocket.core.dispatcher import Dispatcher


class Looper(Dispatcher):
    def __init__(self,
                 capsules: List[Capsule],
                 tag: str="Looper",
                 grad_enabled: bool=True,
                 repeats: int=None,
                 run_every: int=1,
                 statefull: bool=True,
                 accelerator: Accelerator=None,
                 priority: int=1000):
        super().__init__(capsules=capsules,
                         accelerator=accelerator,
                         priority=priority)
        self._statefull = statefull
        self._repeats = repeats or -1
        self._grad_enabled = grad_enabled
        self._run_every = run_every
        self._epoch_idx = 1
        self._batch_idx = 0
        self._tag = tag

    def run_if_needed(method):
        def wrapper(self, attrs: Attributes=None):
            if self._epoch_idx % self._run_every != 0:
                return
            method(self, attrs=attrs)
        return wrapper


    def set(self, attrs: Attributes=None):
        Dispatcher.set(self, attrs=attrs)

        if self._repeats < 0:
            self.infer_repeats()
        
        if self._repeats < 0:
            err = f"{self.__class__.__name__}: infinite loops are not allowed. "
            err += "Please, specify number of repeats."
            raise RuntimeError(err)

    def reset(self, attrs: Attributes=None):
        self._epoch_idx += 1
        Dispatcher.reset(self, attrs=attrs)
        

    @run_if_needed
    def launch(self, attrs: Attributes):
        if attrs.looper is None:
            # if loop state is not user-defined, do it
            attrs.looper = Attributes(repeats=self._repeats,
                                      state=Attributes(),
                                      terminate=False)

        desc = f"{colored(self._tag, 'green')} " 
        desc += f"epoch={self._epoch_idx}, "
        desc += f"grad={self._grad_enabled}"

        status_bar = tqdm(range(self._repeats),
                          initial=0,
                          desc=desc,
                          # Only show the progress bar once on each machine.
                          disable=not self._accelerator.is_local_main_process)
        
        for _ in range(self._repeats):
            # reset batch after each iteration
            attrs.batch = None
            # set gradient mode
            with torch.set_grad_enabled(self._grad_enabled):
                # dispatch single forward pass
                Dispatcher.launch(self, attrs)
            # if loop is completed, exit cycle
            if attrs.looper.terminate:
                break
            status_bar.set_postfix(attrs.looper.state)
            status_bar.update(1)
    
        # clean loops info
        del attrs.looper

        self._batch_idx = 0
        self._repeats = -1


    def state_dict(self):
        return Attributes(epoch_idx=self._epoch_idx,
                          batch_idx=self._batch_idx)
    
    def load_state_dict(self, state):
        self._epoch_idx = state.epoch_idx
        self._batch_idx = state.batch_idx

    def guard(self, capsules: List[Capsule]):
        super().guard(capsules)
        for capsule in capsules:
            if isinstance(capsule, Looper):
                err = f"{self.__class__.__name__}: internal loopers are not allowed."
                raise RuntimeError(err)

    def infer_repeats(self):
        for capsule in self._capsules:
            # infer repeats from dataset only
            if isinstance(capsule, Dataset):
                self._repeats += capsule._total
        message = f"{self.__class__.__name__} infered {self._repeats} repeats."
        
        self._logger.info(message)
