import copy
import collections
from typing import List

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes
from rocket.core.dispatcher import Dispatcher


class Meter(Dispatcher):
    def __init__(self, 
                 capsules: List[Capsule], 
                 keys: List,
                 accelerator: Accelerator = None, 
                 priority=1000) -> None:
        super().__init__(capsules=capsules, 
                         accelerator=accelerator, 
                         priority=priority)
        self._keys = keys
        
    def launch(self, attrs: Attributes = None):
        # nothing to do if empty data flow passed
        if attrs is None or attrs.batch is None:
            return
        
        # main logic
        try:
            inputs = [attrs.batch[key] for key in self._keys]
            gathered_inputs = self._accelerator.gather_for_metrics(inputs)
            # store gathered data into batch for further processing
            # be ware of capsule ordering, it may happen that not all of 
            # the original batch keys will be gathered that leads to 
            # different batch sizes in different keys of the batch

            batch, batch_type = attrs.batch, type(attrs.batch)
            if isinstance(batch, collections.abc.Mapping):
                try:
                    if isinstance(batch, collections.abc.MutableMapping):
                        # The mapping type may have extra properties, so we can't just
                        # use `type(data)(...)` to create the new mapping.
                        # Create a clone and update it if the mapping type is mutable.
                        clone = copy.copy(batch)
                        # overwrite gathered keys
                        for key, value in zip(self._keys, gathered_inputs):
                            clone.update({key: value})
                        attrs.batch = clone
                    else:
                        # manual copy since original is immutable and we cant copy and paste
                        clone = {key: value for key, value in batch.items()}
                        # overwrite gathered keys
                        for key, value in zip(self._keys, gathered_inputs):
                            clone.update({key: value})
                        # cast to original type
                        attrs.batch = batch_type(clone)
                except TypeError:
                    # The mapping type may not support `copy()` / `update(mapping)`
                    # or `__init__(iterable)`.
                    # unknown mapping, use default dict
                    clone = {key: value for key, value in batch.items()}
                    # overwrite gathered keys
                    for key, value in zip(self._keys, gathered_inputs):
                        clone.update({key: value})
                    attrs.batch = Attributes(clone)
            
            elif isinstance(batch, collections.abc.Sequence):
                try:
                    if isinstance(batch, collections.abc.MutableSequence):
                        # The sequence type may have extra properties, so we can't just
                        # use `type(data)(...)` to create the new sequence.
                        # Create a clone and update it if the sequence type is mutable.
                        clone = copy.copy(batch)  # type: ignore[arg-type]
                        # overwrite gathered keys
                        for i, sample in zip(self._keys, gathered_inputs):
                            clone[i] = sample
                        attrs.batch = clone
                    else:
                        clone = [sample for sample in batch]
                        # overwrite gathered keys
                        for i, sample in zip(self._keys, gathered_inputs):
                            clone[i] = sample
                        attrs.batch = batch_type(clone)
                except TypeError:
                    # The sequence type may not support `copy()` / `__setitem__(index, item)`
                    # or `__init__(iterable)` (e.g., `range`).
                    clone = [sample for sample in batch]
                    # overwrite gathered keys
                    for i, sample in zip(self._keys, gathered_inputs):
                        clone[i] = sample
                    attrs.batch = clone
        except:
            err = f"{self.__class__.__name__}: some keys are not found in batch"
            raise RuntimeError(err)
        
        Dispatcher.launch(self, attrs=attrs)
        

class Metric(Capsule):
    def __init__(self,
                 accelerator: Accelerator=None, 
                 priority: int=1000) -> None:
        super().__init__(accelerator=accelerator,
                         priority=priority)

    def launch(self, attrs: Attributes = None):
        raise NotImplementedError(f"{self.__class__.__name__}: " + \
                                  "metric should implement launch()")
        
    def reset(self, attrs: Attributes = None):
        raise NotImplementedError(f"{self.__class__.__name__}: " + \
                                  "metric should implement reset()")
        