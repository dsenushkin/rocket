import copy
import torch
import collections

from torch.utils.data._utils.collate import collate, collate_tensor_fn

from collections import defaultdict
from typing import Mapping, Sequence, List, Union, Tuple, Dict, Optional, Type, Callable



MapType = Dict[Union[Type, Tuple[Type, ...]], Callable]



def collate_not_tensor_fn(batch, *, collate_fn_map: Optional[MapType] = None):
    return batch

default_collate_fn_map = {torch.Tensor: collate_tensor_fn}
default_collate_fn_map[str] = collate_not_tensor_fn
default_collate_fn_map[float] = collate_not_tensor_fn
default_collate_fn_map[int] = collate_not_tensor_fn
default_collate_fn_map[tuple] = collate_not_tensor_fn


def default_collate(batch):
    return collate(batch, collate_fn_map=default_collate_fn_map)



def move_str_fn(batch, device, *, move_fn_map: Optional[MapType] = None):
    return batch

def move_tensor_fn(batch, device, *, move_fn_map: Optional[MapType] = None):
    return batch.to(device)

def move_module_fn(batch, device, *, move_fn_map: Optional[MapType] = None):
    return batch.to(device)

def move(batch, device, *, move_fn_map: Optional[MapType] = None):
    batch_type = type(batch)

    if move_fn_map is not None:
        if batch_type in move_fn_map:
            return move_fn_map[batch_type](batch, device, move_fn_map=move_fn_map)

        for move_type in move_fn_map:
            if isinstance(batch, move_type):
                return move_fn_map[move_type](batch, device, move_fn_map=move_fn_map)

    if isinstance(batch, collections.abc.Mapping):
        try:
            if isinstance(batch, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(batch)
                clone.update({key: move(batch[key], device, move_fn_map=move_fn_map) for key in batch})
                return clone
            else:
                return batch_type({key: move(batch[key], device, move_fn_map=move_fn_map) for key in batch})
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {key: move(batch[key], device, move_fn_map=move_fn_map) for key in batch}
        
    # elif isinstance(batch, tuple) and hasattr(batch, '_fields'):  # namedtuple
    #     return batch_type(*(move(samples, move_fn_map=move_fn_map) for samples in zip(*batch)))
    
    elif isinstance(batch, collections.abc.Sequence):
        try:
            if isinstance(batch, collections.abc.MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(batch)  # type: ignore[arg-type]
                for i, sample in enumerate(batch):
                    clone[i] = move(sample, device, move_fn_map=move_fn_map)
                return clone
            else:
                return batch_type([move(sample, device, move_fn_map=move_fn_map) for sample in batch])
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [move(sample, device, move_fn_map=move_fn_map) for sample in batch]

    raise TypeError(f"{batch_type} move error.")

default_move_fn_map = {torch.Tensor: move_tensor_fn}
default_move_fn_map[torch.nn.Module] = move_module_fn
default_move_fn_map[str] = move_str_fn
default_move_fn_map[float] = move_str_fn
default_move_fn_map[int] = move_str_fn
# default_move_fn_map[None] = move_str_fn

def default_move(batch, device):
    return move(batch, device, move_fn_map=default_move_fn_map)
