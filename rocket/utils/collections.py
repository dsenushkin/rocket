# https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py
import copy
import collections.abc
from typing import Callable


def is_collection(x):
    return isinstance(x, collections.abc.Mapping) or isinstance(x, collections.abc.Sequence)


def _apply_to_mapping(container, fn: Callable, **kwargs):
    return {key: fn(container[key], key=key, **kwargs) for key in container}


def _apply_to_sequence(container, fn: Callable, **kwargs):
    return [fn(sample, key=i, **kwargs) for i, sample in enumerate(container)]


def _apply_to_sequence_copy(container, fn: Callable, **kwargs):
    clone = copy.copy(container)  # type: ignore[arg-type]
    for i, sample in enumerate(container):
        clone[i] = fn(sample, key=i, **kwargs)
    return clone


def apply_to_mapping(container: collections.abc.Mapping, fn: Callable, **kwargs):
    new_mapping = _apply_to_mapping(container, fn, **kwargs)
    try:
        if isinstance(container, collections.abc.MutableMapping):
            # The mapping may contain additional properties in the class.
            # Therefore, we first copy, and then update only the keys.
            # Updating is allowed only for mutable mappings
            clone = copy.copy(container)
            clone.update(new_mapping)
            return clone
        else:
            return type(container)(new_mapping)
    except TypeError:
        # The mapping doesn't have .copy(), .update() methods or
        # __init__(iterable)
        # Using a default dictionary, possible data loss
        return new_mapping


def apply_to_sequence(container: collections.abc.Sequence, fn: Callable, **kwargs):
    try:
        if isinstance(container, collections.abc.MutableSequence):
            # Lists may contain additional properties in the class.
            # Therefore, we first copy, and then update by indices.
            # Updating is allowed only for mutable lists
            return _apply_to_sequence_copy(container, fn, **kwargs)
        else:
            return type(container)(_apply_to_sequence(container, fn, **kwargs))
    except TypeError:
        # The list doesn't have .copy(), .update() methods or
        # __init__(iterable)
        # Using a default list, possible data loss
        return _apply_to_sequence(container, fn, **kwargs)


def apply_to_collection(
    container: collections.abc.Mapping | collections.abc.Sequence,
    fn: Callable,
    **kwargs
) -> collections.abc.Mapping | collections.abc.Sequence:
    if isinstance(container, collections.abc.Mapping):
        return apply_to_mapping(container, fn, **kwargs)
    elif isinstance(container, collections.abc.Sequence):
        return apply_to_sequence(container, fn, **kwargs)
    else:
        raise TypeError("{} is not a collection.".format(type(container)))
