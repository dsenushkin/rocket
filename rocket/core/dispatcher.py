from operator import attrgetter

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes, Events


class Dispatcher(Capsule):
    """Класс-интерфейс для обработчиков событий.
    По умолчани реализует последовательный вызов обработчиков событий
    для каждой капсулы, которую содержит. Все обработчики вызываются
    в прямом порядке за исключением :code:`.destroy(attrs)`,
    для которого задан обратный порядок вызова.

    Parameters
    ----------
    capsules : list[Capsule]
        Капсулы. Список обработчиков событый.
    accelerator : Accelerator | None, optional, default = None
        Объект класса accelerate.Accelerator.
    priority : int, optional, default = 1000
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        capsules: list[Capsule],
        accelerator: Accelerator | None = None,
        priority: int = 1000
    ) -> None:
        super().__init__(accelerator=accelerator,
                         statefull=False,
                         priority=priority)

        self.guard(capsules)
        self._capsules = sorted(capsules,
                                key=attrgetter("_priority"),
                                reverse=True)
        self.accelerate(accelerator)

    def setup(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SETUP`.
        Вызывает метод :code:`.setup(attrs)` у всех капсул,
        которые содержит. Порядок вызова - прямой последовательный.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.setup(self, attrs=attrs)

        for capsule in self._capsules:
            capsule.dispatch(Events.SETUP, attrs)

    def destroy(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.DESTROY`.
        Вызывает метод :code:`.destroy(attrs)` у всех капсул,
        которые содержит. Порядок вызова - обратный последовательный.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        for capsule in reversed(self._capsules):
            capsule.dispatch(Events.DESTROY, attrs)

        Capsule.destroy(self, attrs=attrs)

    def set(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SET`.
        Вызывает метод :code:`.set(attrs)` у всех капсул,
        которые содержит. Порядок вызова - прямой последовательный.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.set(self, attrs=attrs)

        for capsule in self._capsules:
            capsule.dispatch(Events.SET, attrs)

    def reset(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.RESET`.
        Вызывает метод :code:`.reset(attrs)` у всех капсул,
        которые содержит. Порядок вызова - прямой последовательный.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.reset(self, attrs=attrs)

        for capsule in self._capsules:
            capsule.dispatch(Events.RESET, attrs)

    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH`.
        Вызывает метод :code:`.launch(attrs)` у всех капсул,
        которые содержит. Порядок вызова - прямой последовательный.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.launch(self, attrs=attrs)

        for capsule in self._capsules:
            capsule.dispatch(Events.LAUNCH, attrs)

    def accelerate(self, accelerator: Accelerator):
        """Метод инициализации акселератора.
        Задает акселератор для всех капсул, которые содержит.

        Parameters
        ----------
        accelerator : Accelerator
            Объект класса accelerate.Accelerator.
        """
        Capsule.accelerate(self, accelerator)

        for capsule in self._capsules:
            capsule.accelerate(accelerator)
    
    def clear(self):
        """Освобождает ресуры.
        """
        Capsule.clear(self)

        for capsule in self._capsules:
            capsule.clear()

    def guard(self, capsules: list[Capsule]):
        """Метод проверки типа переданных капсул.

        Parameters
        ----------
        capsules : list[Capsule]
            Список объектов, производных от :code:`Capsule`

        Raises
        ------
        ValueError
            В списке есть объекты, неунаследованные от :code:`Capsule`.
        """
        for capsule in capsules:
            if not isinstance(capsule, Capsule):
                err = f"{self.__class__.__name__} got invalid capsule."
                raise ValueError(err)

    def __repr__(self) -> str:
        tabs = " " * 4

        def reformat(value):
            return str(value).replace("\n", f"\n{tabs*2}")

        attrs = f"\n{tabs}".join(
            f"{key}={reformat(value)}" 
            for key, value in self.__dict__.items() if key != "_capsules"
        )

        caps = "\n".join(str(cap) for cap in self._capsules)
        caps = caps.replace("\n", f"\n{tabs}")

        caps = f"\n_capsules=[\n{tabs}{caps}\n]"
        caps = caps.replace("\n", f"\n{tabs}")
        attrs += caps
        return f"{self.__class__.__name__}(\n{tabs}{attrs}\n)"
