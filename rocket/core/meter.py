import torch
from typing import List

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes
from rocket.core.dispatcher import Dispatcher
from rocket.utils.collections import apply_to_collection


def rebuild_batch(lookup_table):
    def fn(value, key, **kwargs):
        # если ключ есть - модифицируем
        return lookup_table.get(key, value)
    return fn


class Meter(Dispatcher):
    def __init__(
        self,
        capsules: List[Capsule],
        keys: List,
        accelerator: Accelerator | None = None,
        priority=1000
    ) -> None:
        super().__init__(capsules=capsules,
                         accelerator=accelerator,
                         priority=priority)
        self._keys = sorted(keys)

    def launch(self, attrs: Attributes | None = None):
        # ничего не делает, если буфер пуст
        if attrs is None or attrs.batch is None:
            return

        if torch.is_grad_enabled():
            return

        # собираем значение на каждом процессе
        values = list()
        for key in self._keys:
            values += [attrs.batch[key]]

        # отправляем на глобальный хост для расчета метрик
        gathered_values = self._accelerator.gather_for_metrics(values)

        # Hashtable для быстрого поиска по ключу
        lookup_table = {
            k: v for k, v in zip(self._keys, gathered_values)
        }

        # реформируем батч по нужным ключам
        attrs.batch = apply_to_collection(
            attrs.batch, rebuild_batch(lookup_table)
        )

        Dispatcher.launch(self, attrs=attrs)


class Metric(Capsule):
    def __init__(self,
                 accelerator: Accelerator | None = None,
                 priority: int = 1000) -> None:
        super().__init__(accelerator=accelerator,
                         priority=priority)

    def set(self, attrs: Attributes | None = None):
        Capsule.set(self, attrs)
        self._step = attrs.launcher.epoch_idx

    def launch(self, attrs: Attributes | None = None):
        raise NotImplementedError(f"{self.__class__.__name__}: " +
                                  "metric should implement launch()")

    def reset(self, attrs: Attributes | None = None):
        raise NotImplementedError(f"{self.__class__.__name__}: " +
                                  "metric should implement reset()")
