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
                # Маппинг может содержать дополнительные свойства в классе.
                # Поэтому сначала копируем, а потом обновляем только ключи.
                # Обновление разрешено только для изменяемых маппингов
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
            # У маппинга нет методов .copy(), .update() или __init__(iterable)
            # Используем словарь по умолчанию, возможна потеря данных
            return dict(
                {
                    key: fn(container[key], key=key, **kwargs)
                    for key in container
                }
            )

    elif isinstance(container, collections.abc.Sequence):
        try:
            if isinstance(container, collections.abc.MutableSequence):
                # Списки могут содержать дополнительные свойства в классе.
                # Поэтому сначала копируем, а потом обновляем по индексам.
                # Обновление разрешено только для изменяемых списков
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
            # У списка нет методов .copy(), .update() или __init__(iterable)
            # Используем список по умолчанию, возможна потеря данных
            return list(
                [
                    fn(sample, key=i, **kwargs)
                    for i, sample in enumerate(container)
                ]
            )
    raise TypeError(f"{BTYPE} is not a collection.")
