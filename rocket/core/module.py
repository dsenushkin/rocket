import sys
import torch

from contextlib import contextmanager

from accelerate import Accelerator

from rocket.core.dispatcher import Dispatcher
from rocket.core.capsule import Capsule, Attributes
from rocket.utils.torch import torch_move


class Module(Dispatcher):
    """Основной класс-обертка для torch.nn.Module.
    Предназначен для интеграции оборачиваемого модуля в пакет rocket.
    Обертка скрывает операции с модулем необходимые для распределенного
    обучения от пользователя. Класс содежрит капсулы, которые должны
    должны выполнятся в контексте модуля. 

    Parameters
    ----------
    module : torch.nn.Module
        Модуль торча, который оборачивается
    capsules : list[Capsule], optional, default = []
        Список капсул, которые выполняются для этого модуля.
    priority : int, optional, default = 1000
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        module: torch.nn.Module,
        capsules: list[Capsule] = [],   # suppose to include
                                        # losses, optimizers,
                                        # schedulers, postprocessors
        accelerator: Accelerator | None = None,
        priority: int = 1000
    ) -> None:
        super().__init__(capsules=capsules,
                         accelerator=accelerator,
                         priority=priority)
        self._module = module

    def setup(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SETUP`.
        Регистрирует модуль в акселераторе, оборачивает его для
        распределенного обучения и переносит веса на вычислитель.
        Для всех внутренних капсул вызывает :code:`.set(attrs)`.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.

        Raises
        ------
        RuntimeError
            Один и тот же модуль зарегистрирован дважды.
        """
        self.check_accelerator()
        # если модуль уже зарегистрирован, этот флаг = True
        registered = False
        # проверяем на дублирование перед регистрацией.
        for model in self._accelerator._models:
            # пропускаем несовпадения
            if self._module is not model:
                continue
            # нашли два, бросаем исключение
            if registered:
                err = f"{self.__class__.__name__}: "
                err += "same module has been registered twice. "
                raise RuntimeError(err)
            # нашли один, берем его
            registered = True
            self._module = model

        # ничего не нашли, регистрируем
        if not registered:
            # перемещаем на девайс в ручном режиме
            self._module = torch_move(self._module, self._accelerator.device)
            # оброачиваем в акселератор
            self._module = self._accelerator.prepare(self._module)

        Dispatcher.setup(self, attrs)

    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH`.
        Метод осуществляет прямой проход по модели.
        Входные данные - attrs.batch. Затем вызывает все внутренние капсулы.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        # модуль работает с батчем из глобального буфера
        if attrs is None or attrs.batch is None:
            return

        # train/eval моды. Ориентируется по градиенту.
        if torch.is_grad_enabled():
            # training mode
            self._module.train()
        else:
            # eval mode
            self._module.eval()

        # вызов прямого прохода в своем контексте
        # with self._accelerator.accumulate(self._module):
        with self.runner():
            attrs.batch = self._module.forward(attrs.batch)
            # вызываем другие капсулы,
            # такие как лоссы, оптимизаторы, планировщики
            Dispatcher.launch(self, attrs=attrs)

    def destroy(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.DESTROY`.
        Удаляет модуль из акселератора. Вызывает деструкторы внутренних капсул.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        _id = None
        # ищем совпадения с текущим
        for id, model in enumerate(self._accelerator._models):
            # пропускаем несовпадения
            if model is not self._module:
                continue
            # нашли
            _id = id
            break
        # удаляем
        if _id is not None:
            self._accelerator._models.pop(_id)

        Dispatcher.destroy(self, attrs=attrs)

    @contextmanager
    def runner(self):
        """Контекст выполнения forward.
        Поддерживает аккумуляцию градиента по шагам и автоприведение типов
        для mixed_precision обучения.
        """
        autocast_ctx = self._accelerator.autocast()
        accumulate_ctx = self._accelerator.accumulate(self._module)

        try:
            autocast_ctx.__enter__()
            accumulate_ctx.__enter__()
            yield
        finally:
            autocast_ctx.__exit__(*sys.exc_info())
            accumulate_ctx.__exit__(*sys.exc_info())
