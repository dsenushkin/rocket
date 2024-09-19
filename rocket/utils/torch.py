import torch
import collections

from torch.utils.data._utils.collate import collate, collate_tensor_fn  # type: ignore # noqa E501
from rocket.utils.collections import apply_to_collection

from typing import Dict, Type, Callable


MapType = Dict[Type, Callable]

BUILTIN_TYPES = [int, float, str, bool, complex, bytes]

# By default, we don't combine lists of any types into tensors
# This behavior differs from that specified in torch
# There, all types that can be converted to a tensor are combined
def _no_collate(batch, *, collate_fn_map: MapType | None = None):   # noqa E302
    return batch

# Factory for defaultdict
def _no_collate_factory():  # noqa E302
    return _no_collate

# Handler table
COLLATE_MAPPINGS = collections.defaultdict(_no_collate_factory) # noqa E302
COLLATE_MAPPINGS[torch.Tensor] = collate_tensor_fn

# We only redefined the mapping of handlers by types
# Everything else is done using torch tools
def torch_collate(batch):   # noqa E302
    # Initialize handler for standard types through the factory
    if type(batch) in BUILTIN_TYPES:
        COLLATE_MAPPINGS[type(batch)]
    return collate(batch, collate_fn_map=COLLATE_MAPPINGS)





# Custom wrapper over torch methods for transferring to device # noqa E302
# By default, the wrapper should not do anything with non-torch types
def _no_move(batch, device, *, move_fn_map: MapType | None = None): # noqa E302
    return batch

# Factory for defaultdict
def _no_move_factory(): # noqa E302
    return _no_move

# Wrapper over torch's .to
def _move_to(batch, device, *, move_fn_map: MapType | None = None):  # noqa E302
    return batch.to(device)

# Handler table
MOVE_MAPPINGS = collections.defaultdict(_no_move_factory)   # noqa E302
MOVE_MAPPINGS[torch.Tensor] = _move_to
MOVE_MAPPINGS[torch.nn.Module] = _move_to

# Process batches with the appropriate handler
def move(batch, device, *, move_fn_map: MapType | None = None, **kwargs): # noqa E302
    BTYPE = type(batch)

    if move_fn_map is not None:
        # Check for direct type correspondence
        if BTYPE in move_fn_map:
            return move_fn_map[BTYPE](
                batch, device, move_fn_map=move_fn_map
            )

        # Check for inheritance from specified types
        for move_type in move_fn_map:
            if isinstance(batch, move_type):
                return move_fn_map[move_type](
                    batch, device, move_fn_map=move_fn_map
                )

    return apply_to_collection(
        batch, move, device=device, move_fn_map=move_fn_map
    )

# Method available for use from outside
def torch_move(batch, device):  # noqa E302
    if type(batch) in BUILTIN_TYPES:
        # Initialize handler for standard types through the factory
        MOVE_MAPPINGS[type(batch)]
    return move(batch, device, move_fn_map=MOVE_MAPPINGS)


def register_move_hook(dtype: type, hook: Callable) -> None:
    if not isinstance(type(dtype), type):
        raise RuntimeError("The provided dtype is not a type.")
    MOVE_MAPPINGS[dtype] = hook


def register_default_move_hook(dtype: type) -> None:
    register_move_hook(dtype=dtype, hook=_move_to)
