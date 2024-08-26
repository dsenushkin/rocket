import os

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes


class Checkpointer(Capsule):
    """Чекпоинтер.
    Этот класс управляет процессом сохранения состояния процесса.
    Средствами акселератора сохраняет модели, оптимизаторы, шедулеры,
    даталоадеры и все состояния капсул, которые были зарегистрированы
    в акселераторе.

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
                    rocket.Checkpointer(output_dir="./logs/",
                        overwrite=True,
                        save_every=50)
                ]),
            ],
            num_epochs=4
        )

    Parameters
    ----------
    output_dir : str
        Путь до директории, где будут сохраняться веса состояния.
    resume : str | None, optional, default = None
        Путь до директории, где лежат веса состояния для восстановления.
    strict : bool, optional, default = True
        Флаг загрузки состояния капсул. По умолчанию пытаемся восстанавливать.
    save_every : int | None, optional, default = None
        Периодичность сохранения сотояния. Отсчитывается в количествах
        вызовов обработчика события :code:`Events.LAUNCH`.
    overwrite : bool, optional, default = True
        Флаг записи весов. Перезаписывает существующие логи при конфликтах.
    statefull : bool, optional, default = True
        Флаг состояни капсулы чекпоинтера. По умолчанию содержит состояние.
    accelerator : Accelerator | None, optional, default = None
        Объекта класса accelerate.Accelerator.
    priority : int, optional, default = 100
        Приоритет вызова обработчиков событий в очереди.
        Чем меньше значение, тем выше приоритет.
    """
    def __init__(
        self,
        output_dir: str,
        resume: str | None = None,
        strict: bool = True,
        save_every: int | None = None,
        overwrite: bool = True,
        statefull: bool = True,
        accelerator: Accelerator | None = None,
        priority: int = 100
    ) -> None:
        super().__init__(accelerator=accelerator,
                         statefull=statefull,
                         priority=priority)
        self._save_every = save_every or -1
        self._output_dir = output_dir
        self._resume = resume
        self._strict = strict
        self._overwrite = overwrite
        self._iter_idx = 0
        # state for distributed
        self._num_procs = 1
        self._num_nodes = 1

    def setup(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.SETUP`.
        Если капсула имеет состояние, то регистрирует ее в акселераторе.
        Дополнительно восстанавливает состояние из чекпоинта, если необходимо.
        В strict=True режиме восстанавливает все состояния капсул в пайплайне.
        Необходимо учитывать distributed контекст, в котором запускался
        чекпоинт, из которого восстанавливается состояние. Контексты должны
        быть идентичны, иначе восстановление стостояний как минимум 
        даталоадеров будет некорректным.
        В strict=False режиме восстановление возможно только 
        без распределенного обучения.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.setup(self, attrs=attrs)

        self._resume = attrs.checkpointer.resume
        self._strict = attrs.checkpointer.strict

        if self._resume is not None:
            if not self._strict:
                # strict mode регулирует загрузку стейтов капсул
                # strict = False -> не загружаем капсулы
                # чистим буфер акселератора перед загрузкой весов
                custom_objects = self._accelerator._custom_objects
                self._accelerator._custom_objects = []
                try:
                    # Игнорируем исключение вызванное разным количеством
                    # обнаруженных весов и зарегистрированных объектов.
                    self._accelerator.load_state(self._resume)
                except RuntimeError:
                    # возвращаем назад, чтобы иметь возможность логировать
                    self._accelerator._custom_objects = custom_objects
            else:
                # strict = True -> восстанавливаем полное состояние
                self._accelerator.load_state(self._resume)

            if self._num_procs != attrs.launcher.num_procs or \
                    self._num_nodes != attrs.launcher.num_nodes:
                raise RuntimeError("You need to resume your training in " +
                                   "the exact same distributed setup.")

        self._num_procs = attrs.launcher.num_procs
        self._num_nodes = attrs.launcher.num_nodes

    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH`.
        С заданной периодичностью выполняет сохраниение состояния пайплайна.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.

        Raises
        ------
        RuntimeError
            Неразрешенная попытка переписать существующую директорию.
        """
        # вызываем для возможного дополнения функциональности по умолчанию
        Capsule.launch(self, attrs=attrs)

        if not self._accelerator.is_main_process:
            return

        # отрицательный период, значит не делаем ничего
        if self._save_every < 0:
            return

        # сохранение всех зарегистрированных объектов
        if (self._iter_idx + 1) % self._save_every == 0:

            output_dir = os.path.join(self._output_dir, str(self._iter_idx))
            if not self._overwrite and os.path.exists(output_dir):
                err = f"{self.__class__.__name__}: overwrite is set to False. "
                err += f"{output_dir}"
                raise RuntimeError(err)

            self._accelerator.save_state(output_dir=output_dir)
            self._logger.info(f"{self.__class__.__name__}: saved {output_dir}")
            print(attrs.launcher.epoch_idx, self._iter_idx)
        self._iter_idx += 1

    def state_dict(self):
        # +1, т.к. launch сохраняет предыдущий индекс
        return Attributes(iter_idx=self._iter_idx + 1,
                          num_procs=self._num_procs,
                          num_nodes=self._num_nodes)

    def load_state_dict(self, state):
        self._iter_idx = state.iter_idx
        self._num_procs = state.num_procs
        self._num_nodes = state.num_nodes
