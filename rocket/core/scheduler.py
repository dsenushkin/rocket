import torch

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes


class Scheduler(Capsule):
    """Основной класс-обертка планировщика.
    Оборачивает базовый объект для обработки аккумуляции градиента.
    Должен использоваться в контексте rocket.Module.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Объект планировщика
    accelerator : Accelerator | None, optional, default = None
        Объект класса accelerate.Accelerator.
    priority : int, optional, default = 1000
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        accelerator: Accelerator | None = None,
        priority: int = 1000
    ) -> None:
        super().__init__(accelerator=accelerator,
                         statefull=False,
                         priority=priority)
        self._scheduler = scheduler

    def setup(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SETUP`.
        Регистрирует планировщик в акселераторе.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.

        Raises
        ------
        RuntimeError
            Планировщик зарегистрирован дважды.
        """
        Capsule.setup(self, attrs=attrs)
        # безопасно регистрируем планировщик, если такой уже есть
        # в акселераторе, просто возвращаем его
        # дважды зарегать один планировщик запрещено
        registered = False
        for sched in self._accelerator._schedulers:
            # ищем и пропускаем несовпадения
            if self._scheduler is not sched.scheduler:
                continue
            # нашли дважды, бросаем исключение
            if registered:
                err = f"{self.__class__.__name__}: "
                err += "same scheduler has been registered twice. "
                raise RuntimeError(err)

            # нашли один, возвращаем
            registered = True
            self._scheduler = sched

        # не нашли, регистрируем
        if not registered:
            self._scheduler = self._accelerator.prepare(self._scheduler)

    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH`.
        Вызывается шаг планировщика. Так же публикует состояние для трекера.

        Parameters
        ----------
        attrs : Attributes, optional, default = None
            Глобальный буфер обмена данными.
        """
        # if training is disabled, nothing to do
        if torch.is_grad_enabled():
            self._scheduler.step()

    def destroy(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.DESTROY`.
        Удаляет планировщик из аккселератора

        Parameters
        ----------
        attrs : Attributes, optional, default = None
            Глобальный буфер обмена данными.
        """
        # безопасно удаляем из акселератора
        _id = None
        for id, scheduler in enumerate(self._accelerator._schedulers):
            if scheduler is not self._scheduler:
                continue
            _id = id
            break
        if _id is not None:
            self._accelerator._schedulers.pop(_id)

        Capsule.destroy(self, attrs=attrs)
