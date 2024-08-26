from typing import Iterable

from accelerate import Accelerator

import torch.utils.data

from rocket.core.capsule import Capsule, Attributes
from rocket.utils.torch import torch_collate, torch_move


class Dataset(Capsule):
    """Датасет.
    Этот класс управляет генератором данных в пайплайне. Он считывает
    кусок данных с помощью предоставленного итерируемого объекта, перемещает
    его на нужное устройство и кладет в буфер обмена данными, те в attrs.
    Объект данного класса имеет состояние и может быть сохранен и восстановлен
    из состояния детерминированным образом за счет акселератора.

    Пример:

    .. code-block:: python

        mnist = MNIST(path, ...)
        net = myAwesomeNet(...)
        loss = myAwesomeLoss(...)
        opt = myAwesomeOpt(...)


        launcher = rocket.Launcher([
                rocket.Looper([
                    rocket.Dataset(mnist, batch_size=1024),
                    rocket.Module(net, capsules=[
                        rocket.Loss(objective=loss),
                        rocket.Optimizer(opt),
                    ]),
                ]),
            ],
            num_epochs=4
        )

    Parameters
    ----------
    dataset : Iterable
        Итерируемый объект, считывающий данные с диска.
    statefull : bool, optional, default = True
        Флаг состояния. Датасет по умолчанию имеет состояние.
    accelerator : Accelerator | None, optional, default = None
        Объект класса accelerate.Accelerator.
    priority : int, optional, default = 1000
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        dataset: Iterable,
        statefull: bool = True,
        accelerator: Accelerator | None = None,
        priority: int = 1000,
        **kwargs
    ):
        super().__init__(accelerator=accelerator,
                         statefull=statefull,
                         priority=priority)
        # объект датасета
        self._dataset = dataset
        # дефолтный даталоадер
        self._dataloader = None
        # текущий даталоадер, отличается от дефолтного в случае 
        # восстановления класса из состояния с неполным проходом по иходному
        self._active_dataloader = None
        # итератор, выдает данные по next(self._iterator)
        self._iterator = None

        # pytorch dataloader аргументы
        self._kwargs = kwargs
        # модифицированный collate
        self._kwargs.setdefault('collate_fn', torch_collate)

        # индексация общего размера и текущей итерации по данным
        self._batch_idx = 0
        self._total = 0

    def setup(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SETUP`.
        Если капсула имеет состояние, то регистрирует ее в акселератор
        для возможности ее чекпоинтинга.
        Создает даталоадер и регистрирует его в акселераторе.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.

        Raises
        ------
        RuntimeError
            Датасет зарегистрирован дважды в акселераторе.
        """
        Capsule.setup(self, attrs=attrs)

        registered = False

        # Регистрация даталоадера с проверкой дубликатов.
        for dataloader in self._accelerator._dataloaders:
            if self._dataset is not dataloader.dataset:
                continue

            if registered:
                # Датасет зарегистрирован дважды. Бросаем исключение.
                err = f"{self.__class__.__name__}: "
                err += "same dataset has been registered twice."
                raise RuntimeError(err)

            # Нашли первый даталоадер с таким датасетом, запоминаем его.
            registered = True
            self._dataloader = dataloader

        # Не нашли совпадений, регистрируем
        if not registered:

            self._dataloader = torch.utils.data.DataLoader(
                self._dataset, **self._kwargs
            )
            self._dataloader = self._accelerator.prepare(
                self._dataloader, device_placement=[False]
            )

    def set(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SET`.
        Создает итератор по данным изходя из текущего состояния объекта.
        Поддерживает детерминированное восстановление состояния.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.set(self, attrs=attrs)

        # Восстанавливаем состояние, если оно отлично от дефолтного.
        # Состояние не восстанавливается, если метод вызван в no_grad контексте
        if torch.is_grad_enabled() and self._batch_idx > 0:
            self._active_dataloader = self._accelerator.skip_first_batches(
                self._dataloader, self._batch_idx
            )
        else:
            self._active_dataloader = self._dataloader

        self._total = len(self._active_dataloader)
        self._iterator = iter(self._active_dataloader)

    def reset(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.RESET`.
        Вызывается при полном проходе по данному датасету.
        Обнуляет внутреннее состояние в дефолтное.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.reset(self, attrs=attrs)
        # По результатам полного прохода по данным
        # 1. Обнуляем текущий индекс для возможности повторного прохода
        self._batch_idx = 0
        # 2. Обнуляем общий счетчик, это необходимо для восстановления
        self._total = 0
        # 3. Обнуляем итератор
        self._iterator = None

    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH`.
        Выполняет fetch данных из итератора и кладет в буфер обмена.
        Ничего не делает, если буфер обмена не задан или занят.
        Перезаписывает поля управления циклом в буфере, если цикл задан.
        Если данные закончились attrs.looper.terminate = True.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.launch(self, attrs=attrs)

        # Ничего не делай, если буфер обмена не задан или уже занят.
        if attrs is None or attrs.batch is not None:
            return

        data = next(self._iterator, None)

        if data is None:
            # итератор пустой
            attrs.batch = data

            # если был цикл, голосуем за выход
            if attrs.looper is not None:
                attrs.looper.terminate = True
                return
        else:
            # итератор не пустой
            device = self._accelerator.device

            # перемещаем на устройство и кладем в буфер
            attrs.batch = torch_move(data, device)

            # если был цикл, голосуем за продолжение
            if attrs.looper is not None:
                attrs.looper.terminate = False

            self._batch_idx += 1

    def destroy(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.DESTROY`.
        Удаляет даталоадеры внутри себя и из акселератора.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.destroy(self, attrs=attrs)

        # итератор уже освобожден в reset()
        # освобождаем даталоадеры
        self._dataloader = None
        self._active_dataloader = None

        # чистим акселератор
        _id = None
        for id, dataloader in enumerate(self._accelerator._dataloaders):
            # ищем подходящий даталоадер
            if dataloader is not self._dataloader:
                continue
            _id = id
            break
        # удаляем
        if _id is not None:
            self._accelerator._dataloaders.pop(_id)

    def state_dict(self):
        return Attributes(batch_idx=self._batch_idx)

    def load_state_dict(self, state: Attributes):
        self._batch_idx = state.batch_idx
