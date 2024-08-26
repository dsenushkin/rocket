import torch

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes


class Optimizer(Capsule):
    """Основной класс-обертка оптимизатора.
    Оборачивает базовый объект для обработки аккумуляции градиента
    и масштабирования градиента для mixed_precision. Должен использоваться
    в контексте rocket.Module.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Оптимизатор.
    tag : str, optional, default = "opt"
        Тег оптимизатора. Используется для публикации логов в трекере.
    accelerator : Accelerator | None, optional, default = None
        Объекта класса accelerate.Accelerator.
    priority : int, optional, default = 1000
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        tag: str = "opt",
        accelerator: Accelerator | None = None,
        priority: int = 1000
    ) -> None:
        super().__init__(accelerator=accelerator,
                         statefull=False,
                         priority=priority)
        self._optimizer = optimizer
        self._tag = tag
        self._iter_idx = 0

    def setup(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SETUP`.
        Регистрирует оптимизатор в акселераторе.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.

        Raises
        ------
        RuntimeError
            Оптимизатор зарегистрирован дважды.
        """
        Capsule.setup(self, attrs=attrs)

        registered = False
        # безопасно регистрируем оптимизатор, если такой уже есть
        # в акселераторе, просто возвращаем его
        # дважды зарегать один оптимизатор запрещено
        for optim in self._accelerator._optimizers:
            # ищем и пропускаем несовпадения
            if self._optimizer is not optim.optimizer:
                continue
            # нашли дважды, бросаем исключение
            if registered:
                err = f"{self.__class__.__name__}: "
                err += "same optimizer has been registered twice. "
                raise RuntimeError(err)

            # нашли один, возвращаем
            registered = True
            self._optimizer = optim

        # не нашли, регистрируем
        if not registered:
            self._optimizer = self._accelerator.prepare(self._optimizer)

    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH`.
        Вызывается шаг оптимизатора. Так же публикует состояние для трекера.

        Parameters
        ----------
        attrs : Attributes, optional, default = None
            Глобальный буфер обмена данными.
        """
        # обернутый оптимизатор сам понимает, когда пропускать шаги
        if torch.is_grad_enabled():
            self._optimizer.step()
            self._optimizer.zero_grad()

        # постим состояние оптимизатора для трекера при совершении шага
        if self._accelerator.sync_gradients:
            state = Attributes(
                step=self._iter_idx,
                data={
                    f"{self._tag}.lr.{idx}": group.get("lr")
                    for idx, group in enumerate(self._optimizer.param_groups)
                }
            )
            if attrs.tracker is not None:
                attrs.tracker.scalars.append(state)

            if attrs.looper is not None:
                attrs.looper.state.lr = list(state.data.values())

            self._iter_idx += 1

    def destroy(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.DESTROY`.
        Удаляет оптимизатор из аккселератора

        Parameters
        ----------
        attrs : Attributes, optional, default = None
            Глобальный буфер обмена данными.
        """
        # безопасно удаляем из акселератора
        _id = None
        for id, optimizer in enumerate(self._accelerator._optimizers):
            if optimizer is not self._optimizer:
                continue
            _id = id
            break

        if _id is not None:
            self._accelerator._optimizers.pop(_id)

        Capsule.destroy(self, attrs=attrs)

    def state_dict(self):
        return Attributes(iter_idx=self._iter_idx)

    def load_state_dict(self, state: Attributes):
        self._iter_idx = state.iter_idx
