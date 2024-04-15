import logging
from typing import Iterable

from accelerate import Accelerator

import torch.utils.data

from rocket.core.capsule import Capsule, Attributes
from rocket.utils import default_collate, default_move


class Dataset(Capsule):
    def __init__(self,
                 dataset: Iterable,
                 statefull: bool = True,
                 device_placement: bool = True,
                 accelerator: Accelerator = None,
                 priority: int =1000,
                 **kwargs):
        super().__init__(accelerator=accelerator,
                         statefull=statefull,
                         priority=priority)
        self._dataset = dataset
        self._dataloader = None
        self._active_dataloader = None
        self._iterator = None
        self._device_placement = device_placement
        # dataloader args
        self._kwargs = kwargs
        self._kwargs.update(collate_fn=default_collate)
        # loop indices
        self._batch_idx = 0
        self._total = 0


    def setup(self, attrs: Attributes=None):
        # log setup state
        Capsule.setup(self, attrs=attrs)

        registered = False
        # loop over all registered models
        for dataloader in self._accelerator._dataloaders:
            # skip other models if exist
            if self._dataset is not dataloader.dataset:
                continue
            # if same object found twice, raise exeption
            if registered:
                err = f"{self.__class__.__name__}: "
                err += "same dataset has been registered twice. "
                raise RuntimeError(err)
            # everything is of, get modified module
            registered = True
            self._dataloader = dataloader
        # module not found, register it
        if not registered:
            # default torch dataloader
            self._dataloader = torch.utils.data.DataLoader(self._dataset, 
                                                           **self._kwargs)
            # if distributed, prepare it
            self._dataloader = self._accelerator.prepare(self._dataloader)


    def set(self, attrs: Attributes=None):
        # default debug log
        Capsule.set(self, attrs=attrs)
        # if this dataset is in eval mode, does not resume state
        # if state is default, nothing to do
        if torch.is_grad_enabled() and self._batch_idx > 0:
            self._active_dataloader = self._accelerator.skip_first_batches(
                self._dataloader, self._batch_idx
            )
        else:
            self._active_dataloader = self._dataloader
        # total number of iterations left 
        self._total = len(self._active_dataloader)
        # create iterator
        self._iterator = iter(self._active_dataloader)


    def reset(self, attrs: Attributes=None):
        # default debug log
        Capsule.reset(self, attrs=attrs)
        # at the end of an epoch
        # 1. reset batch counter
        self._batch_idx = 0
        # 2. reset total number
        self._total = 0
        # 3. reset iterator
        self._iterator = None


    def launch(self, attrs: Attributes=None):
        # default debug log
        Capsule.launch(self, attrs=attrs)
        # if no attributes provided or 
        # batch has already been created
        # then nothing to do
        if attrs is None or attrs.batch is not None:
            return
        
        # else try to get it
        data = next(self._iterator, None)
        
        if data is None:
            attrs.batch = data
            
            if attrs.looper is not None:
                attrs.looper.terminate = True
                return
        else:
            # move to device
            # if accelerate is properly defined, use it
            device = self._accelerator.device

            if self._device_placement:
                attrs.batch = default_move(data, device)
            else:
                attrs.batch = data
            
            if attrs.looper is not None:
                # data is provided, continue inner loop
                attrs.looper.terminate = False
            # increase batch counter
            self._batch_idx += 1

    def destroy(self, attrs: Attributes=None):
        # log it for human tracking
        Capsule.destroy(self, attrs=attrs)
        # free dataloader, iterator has been freed in reset()
        self._dataloader = None
        self._active_dataloader = None
        # safe pop from accelerator
        _id = None
        for id, dataloader in enumerate(self._accelerator._dataloaders):
            # skip other modules if exit
            if dataloader is not self._dataloader:
                continue
            _id = id
            break
        # pop it from list
        if _id is not None:
            self._accelerator._dataloaders.pop(_id)


    def state_dict(self):
        # if (len(self._dataloader) - self._batch_idx) > 0:
            # method for register_for_checkpointing, gather state
        return Attributes(batch_idx=self._batch_idx)
        # return Attributes(batch_idx=0)
    
    def load_state_dict(self, state):
        # method for register_for_checkpointing, load state
        self._batch_idx = state.batch_idx
