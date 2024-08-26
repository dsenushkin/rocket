import torch
import collections

from torch.utils.data._utils.collate import collate, collate_tensor_fn
from rocket.utils.collections import apply_to_collection

from typing import Dict, Type, Callable


MapType = Dict[Type, Callable]

BUILTIN_TYPES = [int, float, str, bool, complex, bytes]

# по умолчанию мы не объединяем списки любых типов в тензоры
# это поведение отличается от заданного в torch
# там все типы, которые могут быть преобразованы в тензор, объединяются
def _no_collate(batch, *, collate_fn_map: MapType | None = None):   # noqa E302
    return batch

# фабрика для defaultdict
def _no_collate_factory():  # noqa E302
    return _no_collate

# таблица обработчиков
COLLATE_MAPPINGS = collections.defaultdict(_no_collate_factory) # noqa E302
COLLATE_MAPPINGS[torch.Tensor] = collate_tensor_fn

# мы только перезадали маппинг обработчиков по типам
# все остальное выполняется средствами торча
def torch_collate(batch):   # noqa E302
    # инициализируем обработчик для стандартных типов через фабрику
    if type(batch) in BUILTIN_TYPES:
        COLLATE_MAPPINGS[type(batch)]
    return collate(batch, collate_fn_map=COLLATE_MAPPINGS)





# кастомная обертка над методами торча для переноса на устройство # noqa E302
# по умолчанию обертка не должна ничего делать с не торчевыми типами
def _no_move(batch, device, *, move_fn_map: MapType | None = None): # noqa E302
    return batch

# фабрика для defaultdict
def _no_move_factory(): # noqa E302
    return _no_move

# обертка над торчевым .to
def _move_to(batch, device, *, move_fn_map: MapType | None = None):  # noqa E302
    return batch.to(device)

# таблица обработчиков
MOVE_MAPPINGS = collections.defaultdict(_no_move_factory)   # noqa E302
MOVE_MAPPINGS[torch.Tensor] = _move_to
MOVE_MAPPINGS[torch.nn.Module] = _move_to

# обрабатываем батчи подходящим обработчиком
def move(batch, device, *, move_fn_map: MapType | None = None, **kwargs): # noqa E302
    BTYPE = type(batch)

    if move_fn_map is not None:
        # проверка прямого соответствия типу
        if BTYPE in move_fn_map:
            return move_fn_map[BTYPE](
                batch, device, move_fn_map=move_fn_map
            )

        # проверка наследования от заданных типов
        for move_type in move_fn_map:
            if isinstance(batch, move_type):
                return move_fn_map[move_type](
                    batch, device, move_fn_map=move_fn_map
                )

    return apply_to_collection(
        batch, move, device=device, move_fn_map=move_fn_map
    )

# метод дотупный для использования из вне
def torch_move(batch, device):  # noqa E302
    if type(batch) in BUILTIN_TYPES:
        # инициализируем обработчик для стандартных типов через фабрику
        MOVE_MAPPINGS[type(batch)]
    return move(batch, device, move_fn_map=MOVE_MAPPINGS)
