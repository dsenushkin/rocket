import torch
from typing import Callable
from tqdm import tqdm
from termcolor import colored

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes
from rocket.core.dataset import Dataset
from rocket.core.dispatcher import Dispatcher


class Looper(Dispatcher):
    """Класс управления циклом.
    Реализует полный проход внутри одной эпохи.
    Длина эпохи выводится из количества и длины датасетов
    внутри класса. Класс поддерживает возможность скипа некоторых
    эпох, что регулирует параметр run_every. Также есть опция запуска
    цикла без градиента для валидации.

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
    capsules : list[Capsule]
        Список капсул, формирующих одну эпоху
    tag : str, optional, default = "Looper"
        Тег цикла, нужен для вывода в консоль.
    grad_enabled : bool, optional, default = True
        Флаг градиента, по умолчанию разрешен.
    repeats : int | None, optional, default = None
        Количество повторений в цикле. Если не задан, то производится
        попытка вывести длину из датасетов в списке капсул.
    run_every : int, optional, default = 1
        Параметр пропуска эпох. Если > 1, то цикл будер запускаться
        только при epoch %  run_every == 0.
    statefull : bool, optional, default = True
        Флаг сохранения состояния цикла.
    accelerator : Accelerator | None, optional, default = None
        Объект класса accelerate.Accelerator.
    priority : int, optional, default = 1000
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        capsules: list[Capsule],
        tag: str = "Looper",
        grad_enabled: bool = True,
        repeats: int | None = None,
        run_every: int = 1,
        statefull: bool = True,
        accelerator: Accelerator | None = None,
        priority: int = 1000
    ) -> None:
        super().__init__(capsules=capsules,
                         accelerator=accelerator,
                         priority=priority)
        self._statefull = statefull
        # кол-во повторений, которым оперируют методы класса
        self._repeats = None
        # повторения заданные пользователем, необходимы для set
        self._user_defined_repeats = repeats or None
        self._grad_enabled = grad_enabled
        self._run_every = run_every
        self._iter_idx = 0
        self._tag = tag

    def run_if_needed(method: Callable):
        """Декоратор.
        Пропускает запуск метода, если epoch % run_every != 0

        Parameters
        ----------
        method : Callable
            Оборачиваемый метод
        """
        def wrapper(self, attrs: Attributes | None = None):
            epoch = attrs.launcher.epoch_idx
            if epoch % self._run_every != 0:
                return
            method(self, attrs=attrs)
        return wrapper

    @run_if_needed
    def set(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SET`.
        Вызывает метод :code:`.set(attrs)` у капсул внутри объекта.
        Проверяет валидность задания цикла. Если кол-во повторений
        не задано, пытается заинферить это значение из датасетов
        хранимых в объекте. Инициализирует буфер цикла в глобальном буффере
        по необходимости. Может вызываться не каждую эпоху, см. run_every.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.

        Raises
        ------
        RuntimeError
            Бесконечный цикл не разрешен.
        """
        Dispatcher.set(self, attrs=attrs)

        self._repeats = self._user_defined_repeats

        if self._repeats is None:
            self.infer_repeats()

        if self._repeats is None:
            err = f"{self.__class__.__name__}: "
            err += "infinite loops are not allowed. "
            err += "Please, specify number of repeats."
            raise RuntimeError(err)

        if attrs.looper is None:
            # если буфер цикла не задан пользователем, создаем
            attrs.looper = Attributes(repeats=self._repeats,
                                      state=Attributes(),
                                      terminate=False,
                                      tag=self._tag)

    @run_if_needed
    def reset(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.RESET`.
        Вызывает метод :code:`.reset(attrs)` у капсул внутри объекта.
        Увеличивает счетчик эпох, очищает буфер повторений.
        Может вызываться не каждую эпоху, см. run_every.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Dispatcher.reset(self, attrs=attrs)
        self._repeats = None
        del attrs.looper

    @run_if_needed
    def launch(self, attrs: Attributes | None):
        """Обработчик события :code:`Events.LAUNCH`.
        Итеративно вызывает :code:`.launch(attrs)` у всех капсул внутри себя.
        Поддрерживает no_grad контекст.
        Может вызываться не каждую эпоху, см. run_every.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        epoch_idx = attrs.launcher.epoch_idx

        desc = f"{colored(self._tag, 'green')} "
        desc += f"epoch={epoch_idx}, "
        desc += f"grad={self._grad_enabled}"

        status_bar = tqdm(range(self._repeats),
                          initial=0,
                          desc=desc,
                          # показываем прогресс только на локальном хосте
                          disable=not self._accelerator.is_local_main_process)

        for _ in range(self._repeats):
            # очищаем батч после итерации
            attrs.batch = None
            # выставляем контекст градиента
            with torch.set_grad_enabled(self._grad_enabled):
                # вызываем событие
                Dispatcher.launch(self, attrs)
            # шорткат для выхода из цикла
            # другие капсулы могут выставить terminate
            if attrs.looper.terminate:
                break
            # обновляем статус бар
            status_bar.set_postfix(attrs.looper.state)
            status_bar.update(1)

        self._iter_idx = 0
        self._repeats = -1

    def state_dict(self):
        return Attributes(iter_idx=self._iter_idx)

    def load_state_dict(self, state: Attributes):
        self._iter_idx = state.iter_idx

    def guard(self, capsules: list[Capsule]):
        """Метод проверки капсул.
        Осуществляет проверку на вложеность циклов.
        В текущей реализации вложеные циклы запрещены.

        Parameters
        ----------
        capsules : list[Capsule]
            Список капсул для проверки.

        Raises
        ------
        RuntimeError
            Обнаружен вложеный цикл.
        """
        super().guard(capsules)
        for capsule in capsules:
            if isinstance(capsule, Looper):
                err = f"{self.__class__.__name__}: "
                err += "internal loopers are not allowed."
                raise RuntimeError(err)

    def infer_repeats(self):
        """Вывод количества повторений из датасетов.
        """
        repeats = 0
        for capsule in self._capsules:
            # проверяем наследников датасета, у них есть поле _total.
            if isinstance(capsule, Dataset):
                repeats += capsule._total

        if repeats:
            self._repeats = repeats

        message = f"{self.__class__.__name__} infered {self._repeats} repeats."
        self._logger.info(message)
