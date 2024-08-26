
from accelerate import Accelerator
from accelerate.tracking import GeneralTracker

from rocket.core.capsule import Capsule, Attributes


class Tracker(Capsule):
    """Трекер.
    Этот класс управляет процессом логгирования данных в публичные
    сервисы, например, tensorboard. Опирается на интерфейс логеров,
    предоставляемый классом акселератора. Может записывать табличные
    данные и изображения. Мониторит поля :code:`attrs.tracker.scalars`
    и :code:`attrs.tracker.images`.

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
                    rocket.Tracker(),
                ]),
            ],
            num_epochs=4,
        )

    Parameters
    ----------
    backend : Optional[str], optional, default = "tensorboard"
        Backend API провайдер для логирования. Если акселератор не 
        содержит такого трекера, это класс попытается его создать.
    project : Optional[str], optional, default = "project"
        Название проекта, используется при попытке создания трекера.
    project_config : Optional[dict], optional, default = None
        Конфиг проекта, используется при попытке создания трекера.
    accelerator : Optional[Accelerator], optional, default = None
        Объект класса accelerate.Accelerator.
    priority : Optional[int], optional, default = 200
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        backend: str = "tensorboard",
        tag: str = "exp_0",
        config: dict = None,
        accelerator: Accelerator | None = None,
        priority: int = 200
    ) -> None:
        super().__init__(accelerator=accelerator,
                         priority=priority)
        self._backend = backend
        self._tracker = None
        self._tag = tag
        self._config = config or None

    def setup(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SETUP`.
        Если капсула имеет состояние, то метод регистрирует ее в акселератор
        для возможности чекпоинтинга.
        Метод инициализирует поле _tracker. Если трекер с заданным
        бекэндом уже существует в акселераторе, то используется он.
        В противном случае метод пробует создать его с заданными параметрами.

        Parameters
        ----------
        attrs : Attributes, optional, default = None
            Глобальный буфер обмена данными.

        Raises
        ------
        RuntimeError
            Неудачная попытка создать трекер.
        """
        Capsule.setup(self, attrs=attrs)
        self._tracker = self._accelerator.get_tracker(self._backend)

        if type(self._tracker) == GeneralTracker:   # noqa E721
            wrn = f"Accelerator has not initialized {self._backend}."
            wrn += " Trying to create it..."
            self._logger.warn(wrn)

            try:
                self._accelerator.log_with.append(self._backend)
                self._accelerator.init_trackers(self._tag,
                                                self._config)
            except Exception as e:
                err = f"{self.__class__.__name__} can't create tracker: {e}"
                raise RuntimeError(err)

        self._tracker = self._accelerator.get_tracker(self._backend)

    def set(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SET`.
        Инициализирует поля поля :code:`attrs.tracker.scalars`
        и :code:`attrs.tracker.images` в глобальном буфере.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.set(self, attrs=attrs)
        attrs.tracker = Attributes(scalars=[], images=[])

    def launch(self, attrs: Attributes | None = None):
        """_summary_

        Parameters
        ----------
        attrs : Optional[Attributes], optional, default = None
            _description_.
        """
        Capsule.launch(self, attrs=attrs)
        # Трекер ожидает глобальный буфер и свои поля заданными
        if attrs is None or attrs.tracker is None:
            return

        # если буфер пустой, нечего логгировать
        if not attrs.tracker.images and not attrs.tracker.scalars:
            return

        self.log(attrs.tracker.images, attrs.tracker.scalars)
        attrs.tracker = Attributes(scalars=[], images=[])

    def reset(self, attrs: Attributes = None):
        Capsule.reset(self, attrs=attrs)

        if attrs is None or attrs.tracker is None:
            return

        # если буфер пустой, нечего логгировать
        if not attrs.tracker.images and not attrs.tracker.scalars:
            return

        self.log(attrs.tracker.images, attrs.tracker.scalars)
        del attrs.tracker

    def destroy(self, attrs: Attributes = None):
        del self._tracker
        Capsule.destroy(self, attrs=attrs)

    def log(self, images, scalars):
        # если картинки не пустые
        if images and self._accelerator.is_main_process:
            try:
                for image in images:
                    self._tracker.log_images(image.data, step=image.step)
                    self._logger.debug(
                        f"Successfully logged images to {self._backend}"
                    )
            except Exception as e:
                raise RuntimeError(f"Can't log images: {e}")

        # если скаляры не пустые
        if scalars and self._accelerator.is_main_process:
            try:
                for scalar in scalars:
                    self._tracker.log(scalar.data, step=scalar.step)
                    self._logger.debug(
                        f"Successfully logged scalars to {self._backend}"
                    )
            except Exception as e:
                raise RuntimeError(f"Can't log scalars: {e}")
