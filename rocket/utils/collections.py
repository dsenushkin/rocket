# https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py
import copy
import collections.abc
from typing import Callable


def apply_to_collection(
    container: collections.abc.Mapping | collections.abc.Sequence,
    fn: Callable,
    **kwargs
) -> collections.abc.Mapping | collections.abc.Sequence:
    BTYPE = type(container)

    if isinstance(container, collections.abc.Mapping):
        try:
            if isinstance(container, collections.abc.MutableMapping):
                # The mapping may contain additional properties in the class.
                # Therefore, we first copy, and then update only the keys.
                # Updating is allowed only for mutable mappings
                clone = copy.copy(container)
                clone.update(
                    {
                        key: fn(container[key], key=key, **kwargs)
                        for key in container
                    }
                )
                return clone
            else:
                return BTYPE(
                    {
                        key: fn(container[key], key=key, **kwargs)
                        for key in container
                    }
                )
        except TypeError:
            # The mapping doesn't have .copy(), .update() methods or
            # __init__(iterable)
            # Using a default dictionary, possible data loss
            return dict(
                {
                    key: fn(container[key], key=key, **kwargs)
                    for key in container
                }
            )

    elif isinstance(container, collections.abc.Sequence):
        try:
            if isinstance(container, collections.abc.MutableSequence):
                # Lists may contain additional properties in the class.
                # Therefore, we first copy, and then update by indices.
                # Updating is allowed only for mutable lists
                clone = copy.copy(container)  # type: ignore[arg-type]
                for i, sample in enumerate(container):
                    clone[i] = fn(sample, key=i, **kwargs)
                return clone
            else:
                return BTYPE(
                    [
                        fn(sample, key=i, **kwargs)
                        for i, sample in enumerate(container)
                    ]
                )
        except TypeError:
            # The list doesn't have .copy(), .update() methods or
            # __init__(iterable)
            # Using a default list, possible data loss
            return list(
                [
                    fn(sample, key=i, **kwargs)
                    for i, sample in enumerate(container)
                ]
            )
    raise TypeError(f"{BTYPE} is not a collection.")
