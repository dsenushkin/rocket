import logging
from enum import Enum
from adict import adict

from accelerate import Accelerator
from accelerate.logging import get_logger


# Основной инструмент взаимодействия капсулей.
# Выполняет функцию динамического буфера обмена данными.
# Имеет структуру словаря с dot access к полям.
# [!Важно]
# Не бросает исключения при доступе к несуществующему полю.
# Вместо этого возвращает None.
Attributes = adict


class Events(Enum):
    """Базовый класс события, на которые может реагировать капсула.
    """

    SETUP = "setup"
    DESTROY = "destroy"
    SET = "set"
    RESET = "reset"
    LAUNCH = "launch"


class Capsule:
    """Класс-интерфейс для обработчиков событий.
    Инстанцирование класса разрешено, однако в базовом варианте он не несет
    осмысленной функциональности. Основной сценарий использования -
    через наследование и переопределение методов обработки событий.

    Обработчики событий имеют единую сигнатуру:

    .. code-block:: python

        def setup(self, attrs: Attributes | None = None):
            ...

    Обработчику разрешено модифицировать буфер атрибутов.
    Рекомендуется модифицировать только те поля атрибутов,
    которые создавались текущей капсулой.

    [TODO] Добавить проверки на принадлежность поля капсуле.

    Пример:

    .. code-block:: python

        class MyCapsule(Capsule):
            def __init__(self):
                super().__init__()

            def launch(self, attrs: Attributes | None = None):
                print("Hello, world!")

    Parameters
        ----------
        accelerator : Accelerator | None, optional, default = None
            Объект класса accelerate.Accelerator.
        statefull : bool, optional, default = False
            Флаг состояния. Описывает, будет ли сохранятся состояние
            капсулы чекпоинтером.
        logger : logging.Logger | None, optional, default = None
            Объект класса logging.Logger.
        priority : int, optional, default = 1000
            Приоритет вызова обработчиков событий в очереди.
            Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        statefull: bool = False,
        logger: logging.Logger | None = None,
        priority: int = 1000
    ) -> None:
        super().__init__()
        self._priority = priority
        self._statefull = statefull
        self._accelerator = accelerator or None
        self._logger = logger or get_logger(self.__module__)

    def setup(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SETUP` по умолчанию.
        Если капсула имеет состояние, то регистрирует ее в акселератор
        для возможности ее чекпоинтинга.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        self.check_accelerator()

        if self._statefull:
            # Statefull капсулы регистрируются как произвольные объекты
            # с состоянием с помощью акселератора. Он умеет их сохранять.
            # Он хранит их в in self._accelerator._custom_objects.
            self._accelerator.register_for_checkpointing(self)

        message = f"{self.__class__.__name__} initialized."
        self._logger.debug(message)

    def destroy(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.DESTROY` по умолчанию.
        Если капсула была зарегистрированиа, т.е. имеет состояние,
        то этот метод удаляет капсулу из акселератора.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.

        Raises
        ------
        RuntimeError
            Попытка удалить не тот объект, который был зарегистрирован.
        """
        if self._statefull:
            obj = self._accelerator._custom_objects.pop()

            if obj is not self:
                # Попытка удалить не тот объект, который был регистрирован.
                err = f"{self.__class__.__name__}: "
                err += "illegal destroy request. "
                err += f"{obj.__class__.__name__}"

                raise RuntimeError(err)

        message = f"{self.__class__.__name__} destroyed."
        self._logger.debug(message)

    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH` по умолчанию.
        Ничего не делает.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        return

    def set(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SET` по умолчанию.
        Ничего не делает.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        return

    def reset(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.RESET` по умолчанию.
        Ничего не делает.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        return

    def dispatch(self, event: Events,
                 attrs: Attributes | None = None):
        """Метод вызова обработчика по заданному собитию.

        Parameters
        ----------
        event : Events
            Текущее событие
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        return getattr(self, event.value)(attrs)

    def accelerate(self, accelerator: Accelerator):
        """Метод задания акселератора.

        Parameters
        ----------
        accelerator : Accelerator
            Объект класса accelerate.Accelerator
        """
        self._accelerator = accelerator

    def clear(self):
        del self._accelerator
        self._accelerator = None

    def set_logger(self, logger: logging.Logger):
        """Метод задания логгера.

        Parameters
        ----------
        logger : logging.Logger
            Объект класса logging.Logger
        """
        self._logger = logger

    def check_accelerator(self):
        """Метод проверки акселератора.
        Все капсулы должны иметь инициализированный акселератор перед запускм.

        Raises
        ------
        RuntimeError
            Аккселератор не задан.
        """
        if self._accelerator is None:
            err = f"{self.__class__.__name__}: accelerator is not defined. "
            err += "Please, specify it in __init__ function "
            err += "or set it via .accelerate(accelerator) method."
            raise RuntimeError(err)

    def state_dict():
        """Возвращяет сериализуемое состояние в виде словаря adict.
        """
        pass

    def load_state_dict():
        """Восстанавливает состояние из словаря adict.
        """
        pass

    def __repr__(self) -> str:
        """Метод форматирования для вывода на экран.

        Returns
        -------
        str
            Строка описания.
        """
        tabs = " " * 4

        def reformat(value):
            return str(value).replace("\n", f"\n{tabs*2}")

        attrs = f"\n{tabs}".join(
            f"{key}={reformat(value)}" 
            for key, value in self.__dict__.items()
        )
        return f"{self.__class__.__name__}(\n{tabs}{attrs}\n)"
