from typing import Callable
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import ProjectConfiguration

from rocket.core.dispatcher import Dispatcher
from rocket.core.capsule import Attributes, Capsule


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


class Launcher(Dispatcher):
    """Основной класс, отвечающий за запуск циклов.
    Реализует нужную очередность вызова обработчиков событий в
    методе :code:`.launch(attrs)`.

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
        Список циклов для запуска
    tag : str, optional, default = "rocket"
        Название эксперимента.
    logging_dir : str, optional, default = "./logs"
        Путь до корневой папки с логами.
    mixed_precision : str | None, optional, default = None
        Обучение / инференс смешанной точности.
    gradient_accumulation_steps : int, optional, default = 1
        Кол-во шагов для аккумуляции градиента.
    num_epochs : int, optional, default = 1
        Число эпох. Отражает, сколько раз нужно пройтись по циклам.
    statefull : bool, optional, default = False
        Флаг состояния, по умолчанию класс не содержит состояния.
    accelerator : Accelerator | None, optional, default = None
        Объект класса accelerate.Accelerator.
    """
    def __init__(
        self,
        capsules: list[Capsule],
        tag: str = "rocket",
        logging_dir: str = "./logs",
        mixed_precision: str | None = None,
        gradient_accumulation_steps: int = 1,
        num_procs: int | None = None,
        num_nodes: int | None = None,
        num_epochs: int = 1,
        statefull: bool = False,
        # accelerator: Accelerator | None = None
    ) -> None:
        super().__init__(capsules=capsules,
                         accelerator=None)
        self._num_epochs = num_epochs
        self._epoch_idx = 0
        self._statefull = statefull
        self._num_procs = num_procs
        self._num_nodes = num_nodes
        self._mixed_precision = mixed_precision
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._tag = tag
        self._logging_dir = logging_dir
        # resume params
        self._checkpoint = None
        self._strict = True

    def resume(self, checkpoint, strict=True):
        self._checkpoint = checkpoint
        self._strict = strict

    def setup(self, attrs: Attributes | None = None):
        _accelerator = Accelerator(
            device_placement=True,
            mixed_precision=self._mixed_precision,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            # тут нужно создать конфиг, чтобы
            # можно было трекеры создавать на лету
            project_config=ProjectConfiguration(
                project_dir=self._tag,
                logging_dir=self._logging_dir
            )
        )

        self.accelerate(_accelerator)

        Dispatcher.setup(self, attrs)

    def notebook(method: Callable):
        """Декоратор.
        Позволяет запускать распределенное обучение из jupyter.

        Parameters
        ----------
        method : Callable
            Оборачиваемый метод
        """
        def wrapper(self, attrs: Attributes | None = None):

            attrs = attrs or Attributes()
            # обновляем параметры запуска, если они не заданы вручную
            if attrs.launcher is None:
                attrs.launcher = Attributes()
            # параметры распределенного обучения
            attrs.launcher.setdefault('num_procs', self._num_procs or 1)
            attrs.launcher.setdefault('num_nodes', self._num_nodes or 1)

            if attrs.checkpointer is None:
                attrs.checkpointer = Attributes()
            # параметры загрузки чекпоинта
            attrs.checkpointer.setdefault('resume', self._checkpoint or None)
            attrs.checkpointer.setdefault('strict', self._strict or True)

            if in_notebook():
                notebook_launcher(
                    method,
                    args=(self, attrs),
                    num_processes=attrs.launcher.num_procs,
                    num_nodes=attrs.launcher.num_nodes,
                )
            else:
                method(self, attrs=attrs)
        return wrapper

    def set(self, attrs: Attributes | None = None):
        pass

    def reset(self, attrs: Attributes | None = None):
        pass

    @notebook
    def launch(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.LAUNCH`.
        Запускает выполение пайплайна. Пайплайн начинается с
        вызова :code:`.setup(attrs)` для всех капсул. Затем
        в цикле по эпохам последовательно вызываются :code:`.set(attrs)`,
        :code:`.launch(attrs)`, :code:`reset(attrs)` для каждой капсулы.
        Важно отметить, что для заданной капсулы внутри цикла
        последовательно отрабатывают все три обработчика, и только
        после этого выполняется переход к следующей. Предполагается,
        что капсулами, переданными в этот класс будут пайплайны,
        описывающие разные циклы, например TrainLoop, ValLoop и прч.
        По окончанию цикла по эпохам вызывается :code:`.destroy(attrs)`.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Capsule.launch(self, attrs)

        self.setup(attrs)

        for _epoch in range(self._epoch_idx, self._num_epochs):
            attrs.launcher.epoch_idx = _epoch
            self._epoch_idx = _epoch
            # последовательно отрабатываем циклы
            for capsule in self._capsules:
                capsule.set(attrs)
                capsule.launch(attrs)
                capsule.reset(attrs)

        self.destroy(attrs)

    def destroy(self, attrs: Attributes | None = None):
        """Обработчик события :code:`Events.DESTROY`.
        Вызывает метод :code:`.destroy(attrs)` у всех капсул,
        которые содержит. Порядок вызова - обратный последовательный.
        Дополнительно вызывает :code:`accelerator.end_training()`.

        Parameters
        ----------
        attrs : Attributes | None, optional, default = None
            Глобальный буфер обмена данными.
        """
        Dispatcher.destroy(self, attrs=attrs)
        del attrs.launcher
        del attrs.checkpointer
        # заканчиваем обучение
        self._accelerator.end_training()
        self.clear()

    def state_dict(self):
        return Attributes(epoch_idx=self._epoch_idx)

    def load_state_dict(self, state: Attributes):
        self._epoch_idx = state.epoch_idx
