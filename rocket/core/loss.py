import torch

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes


class Loss(Capsule):
    """Основной класс для расчета лосса и градиента.
    Внутри себя обрабатывает аккумуляцию и multi-gpu агрегацию  градиента.
    Должен использоваться в контексте rocket.Module.

    Parameters
    ----------
    objective : torch.nn.Module
        Модуль отвечающий за расчет лосса. 
        На выходе из него должно быть одно число.
    tag : str, optional, default = "train_loss"
        Тег для трекера. По этому тег Loss будет складывать значения.
    accelerator : Accelerator | None, optional, default = None
        Объект accelerate.Accelerator.
    priority : int, optional, default = 1100#приоритетвышеоптимизатора
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        objective: torch.nn.Module,
        tag: str = "train_loss",
        accelerator: Accelerator | None = None,
        priority: int = 1100          # приоритет выше оптимизатора
    ) -> None:
        super().__init__(accelerator=accelerator,
                         statefull=True,
                         priority=priority)
        self._objective = objective
        self._value = 0.0
        self._tag = tag
        self._step = 0

    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH`.
        Вызывает модуль расчета лосса. Поддерживает аккумуляцию градиента.
        Публикует статистику по лоссу в глобальный буфер для трекера.

        Parameters
        ----------
        attrs : Attributes, optional, default = None
            Глобальный буфер обмена данными.
        """
        # Лоссу нужен непустой буфер и батч для работы
        if attrs is None or attrs.batch is None:
            return

        # Лосс не рассчитывается, если градиент выключен
        if not torch.is_grad_enabled():
            return

        # считаем лосс
        loss = self._objective(attrs.batch)

        # агрегируем с других процессов
        gathered_loss = self._accelerator.gather(loss).mean()
        # учитываем множитель для аккумуляции
        self._value += gathered_loss.item() / \
            self._accelerator.gradient_accumulation_steps

        # аккумуляция прошла, градиенты синхронизированы
        if self._accelerator.sync_gradients:
            # отправляем значение трекеру
            if attrs.tracker is not None:
                state = Attributes(
                    step=self._step,
                    data={self._tag: self._value}
                )
                # attrs.tracker.scalars.update({self._tag: self._value})
                attrs.tracker.scalars.append(state)

            if attrs.looper is not None:
                attrs.looper.state.loss = self._value

            # ресетим буфер для трекера
            self._value = 0.0
            self._step += 1

        # считаем градиент
        self._accelerator.backward(loss)

    def state_dict(self):
        return Attributes(value=self._value, step=self._step)

    def load_state_dict(self, state: Attributes):
        self._value = state.value
        self._step = state.step
