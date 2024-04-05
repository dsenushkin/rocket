import torch

from collections import defaultdict
from typing import Mapping, Sequence, List, Union, Tuple



def apply_sequence(data, fn, **kwargs):
    _data = list()
    for value in data:
        _value = apply(value, fn, **kwargs)
        _data += [_value]
    return type(data)(_data)


def apply_mapping(data, fn, **kwargs):
    _data = defaultdict()
    for key, value in data.items():
        _value = apply(value, fn, **kwargs)
        _data[key] = _value
    return type(data)(_data)


def apply(data, fn, **kwargs):
    if isinstance(data, torch.Tensor):
        return fn(data, **kwargs)
    elif isinstance(data, Mapping):
        return apply_mapping(data, fn, **kwargs)
    elif isinstance(data, (List, Tuple)):
        return apply_sequence(data, fn, **kwargs)
    return fn(data, **kwargs)


def _to_device(data, device):
    _data = data
    if hasattr(data, "to"):
        _data = data.to(device)
    return _data


def move(data, device):
    return apply(data, _to_device, device=device)


def collate_sequence(items: List[Sequence]):
    dim = len(items[0])

    for item in items:
        if not isinstance(item, Sequence):
            raise ValueError(f"Inconsistent item's types: {type(item)}.")

        if len(item) != dim:
            raise ValueError(f"Inconsistent item's lengths.")

    data = list()

    for row in list(zip(*items)):
        if isinstance(row[0], torch.Tensor):
            try:
                row = torch.stack(row, 0)
            except:
                raise RuntimeError(f"Tensors can not be collated.")
        data += [row]

    return data


def collate_mapping(items: List[Mapping]):
    data = defaultdict(list)

    keys = set(items[0].keys())

    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError(f"Inconsistent item's types: {type(item)}.")

        allowed_keys = keys

        for key, value in item.items():
            if key not in allowed_keys:
                raise ValueError(f"Inconsistent item's keys: {key}.")
            allowed_keys = allowed_keys - {key}
            data[key] += [value]

        if allowed_keys:
            raise ValueError(f"Inconsistent item's keys: {allowed_keys}.")

    for key, value in data.items():
        if isinstance(value[0], torch.Tensor):
            try:
                data[key] = torch.stack(value, 0)
            except:
                raise RuntimeError(f"Tensors can not be stacked.")

    return data


def collate(items: List[Union[Sequence, Mapping]]):
    if not items:
        raise RuntimeError("Collate got empty list.")

    if isinstance(items[0], Mapping):
        return collate_mapping(items)
    if isinstance(items[0], Sequence):
        return collate_sequence(items)

    raise RuntimeError(f"Collate got unknown collection: {type(items[0])}")