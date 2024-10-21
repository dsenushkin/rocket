# Copyright (c) 2023 Rocket Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Callable
from typing_extensions import Self
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
    """
    A launcher for managing and executing training pipelines.

    This class extends the Dispatcher class and provides functionality
    for setting up, launching, and managing training processes. It handles
    distributed training, checkpointing, and notebook integration.

    Parameters
    ----------
    capsules : list of Capsule
        List of capsules to be managed by the launcher.
    tag : str, optional
        Project tag. Default is "rocket".
    logging_dir : str, optional
        Logging directory. Default is "./logs".
    mixed_precision : str or None, optional
        Mixed precision mode. Default is None.
    gradient_accumulation_steps : int, optional
        Gradient accumulation steps. Default is 1.
    num_procs : int or None, optional
        Number of processes for distributed training. Default is 1.
    num_nodes : int or None, optional
        Number of nodes for distributed training. Default is 1.
    num_epochs : int, optional
        Number of epochs to run. Default is 1.
    statefull : bool, optional
        Whether the launcher maintains state across runs. Default is False.

    Attributes
    ----------
    _num_epochs : int
        The total number of epochs to run.
    _epoch_idx : int
        The current epoch index.
    _statefull : bool
        Whether the launcher maintains state across runs.
    _num_procs : int
        Number of processes for distributed training.
    _num_nodes : int
        Number of nodes for distributed training.
    _mixed_precision : str or None
        Mixed precision mode.
    _gradient_accumulation_steps : int
        Steps for gradient accumulation.
    _tag : str
        Tag for the project.
    _logging_dir : str
        Directory for logging.
    _resume_from : str or None
        Path to resume training from.
    _load_capsules : bool
        Whether to load capsule states when resuming.
    """

    def __init__(
        self,
        capsules: list[Capsule],
        tag: str = "rocket",
        logging_dir: str = "./logs",
        experiment_versioning: bool = True,
        mixed_precision: str | None = None,
        gradient_accumulation_steps: int = 1,
        num_procs: int | None = 1,
        num_nodes: int | None = 1,
        num_epochs: int = 1,
        statefull: bool = False,
    ) -> None:
        super().__init__(capsules=capsules)
        self._num_epochs = num_epochs
        self._epoch_idx = 0
        self._statefull = statefull
        self._num_procs = num_procs
        self._num_nodes = num_nodes
        self._mixed_precision = mixed_precision
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._tag = tag
        self._logging_dir = logging_dir
        self._experiment_versioning = experiment_versioning
        # resume params
        self._resume_from = None
        self._load_capsules = True

    def _resolve_project_dir(self):
        self._project_dir = os.path.join(self._logging_dir, self._tag)
        if not self._experiment_versioning:
            if os.path.isdir(self._project_dir):
                raise ValueError('Project directory already exists and versioning is switched off.'
                                 'Change experiment name or enable experiment versioning')
        else:
            last_version = -1
            if os.path.isdir(self._project_dir):
                versions = sorted(map(lambda x: int(x[1:]), filter(
                    lambda x: x.startswith('v'),
                    os.listdir(self._project_dir)
                )))
                if len(versions) > 0:
                    last_version = versions[-1]
            self._project_dir = os.path.join(self._logging_dir, self._tag, 'v{}'.format(last_version + 1))

    def setup(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.SETUP` event.

        Sets up the accelerator and initializes the launcher for training.

        This method creates an Accelerator instance with the specified
        configuration, including mixed precision and gradient accumulation
        settings. It also sets up the project configuration for logging.
        After creating the accelerator, it applies it to the launcher and
        calls the parent class's setup method.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        self._resolve_project_dir()
        _accelerator = Accelerator(
            device_placement=True,
            mixed_precision=self._mixed_precision,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            project_config=ProjectConfiguration(
                project_dir=self._project_dir,
            ),
            project_dir=self._project_dir
        )

        self.accelerate(_accelerator)

        self._num_procs = attrs.launcher.num_procs
        self._num_nodes = attrs.launcher.num_nodes

        Dispatcher.setup(self, attrs)

    def notebook(method: Callable) -> Callable:
        """
        A decorator adapting the method for notebook or
        non-notebook environments.

        This decorator wraps the given method to handle both notebook and
        non-notebook execution environments. It sets up necessary attributes
        for distributed training and uses the appropriate launcher based on
        the execution context.

        Parameters
        ----------
        method : Callable
            The method to be wrapped.

        Returns
        -------
        Callable
            A wrapper function handling method execution in both environments.

        Notes
        -----
        - Uses `notebook_launcher` from accelerate if in a notebook.
        - Directly calls the method if not in a notebook.
        - Ensures proper setting of launcher attributes (num_procs, num_nodes).
        """
        def wrapper(self, attrs: Attributes | None = None) -> None:

            attrs = attrs or Attributes()
            # Update launch parameters if not manually set
            if attrs.launcher is None:
                attrs.launcher = Attributes()
            # Set distributed training parameters
            attrs.launcher.setdefault('num_procs', self._num_procs)
            attrs.launcher.setdefault('num_nodes', self._num_nodes)

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

    def set(self, attrs: Attributes | None = None) -> None:
        pass

    def reset(self, attrs: Attributes | None = None) -> None:
        pass

    @notebook
    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.LAUNCH` event.

        This method orchestrates the main train/eval loop, including setup,
        resuming from a checkpoint if applicable, and iterating through
        epochs and capsules.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None
        """
        Capsule.launch(self, attrs)

        self.setup(attrs)
        self._resume(attrs)

        for _epoch in range(self._epoch_idx, self._num_epochs):
            attrs.launcher.epoch_idx = _epoch
            self._epoch_idx = _epoch
            # Sequentially process cycles
            for capsule in self._capsules:
                capsule.set(attrs)
                capsule.launch(attrs)
                capsule.reset(attrs)

        self.destroy(attrs)

    def destroy(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.DESTROY` event.

        This method is responsible for cleaning up resources and ending the
        training process. It removes launcher-specific attributes from the
        global exchange buffer, ends the training process in the accelerator,
        and clears the launcher's state.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None
        """
        Dispatcher.destroy(self, attrs=attrs)
        del attrs.launcher
        self._accelerator.end_training()
        self.clear()

    def _resume(self, attrs: Attributes) -> None:
        """
        Resumes training from a checkpoint if specified.

        This method handles the process of resuming training from a checkpoint.
        It can either load the full state (including capsules) or just the
        model weights, depending on the `_load_capsules` flag. It also ensures
        that the distributed setup matches the one used in the checkpoint.

        Parameters
        ----------
        attrs : Attributes
            The global attributes object containing runtime information.

        Raises
        ------
        RuntimeError
            If the checkpoint loading fails or if the distributed setup
            doesn't match the one used in the checkpoint.

        Returns
        -------
        None

        Notes
        -----
        - The method updates `_num_procs` and `_num_nodes` from the attrs
          object, allowing for potential runtime changes to these values.
        """
        if self._resume_from is not None:
            if not self._load_capsules:
                # not loading capsules = clearing accelerator buffer
                custom_objects = self._accelerator._custom_objects
                self._accelerator._custom_objects = []
                try:
                    # ignore exception caused by different number of
                    # detected weights and registered objects
                    self._accelerator.load_state(self._resume_from)
                except RuntimeError:
                    # restore back to be able to log
                    self._accelerator._custom_objects = custom_objects
            else:
                # trying to restore full state
                try:
                    self._accelerator.load_state(self._resume_from)
                except Exception:
                    raise RuntimeError(
                        "Failed to load state from resume checkpoint. "
                        "Please check if the checkpoint is valid and "
                        "compatible with the current launcher configuration."
                    )
            # num_proc & num_nodes are part of the launcher state
            # resuming is only allowed with identical launch parameters
            if self._num_procs != attrs.launcher.num_procs or \
                    self._num_nodes != attrs.launcher.num_nodes:
                raise RuntimeError("You need to resume your training in "
                                   "the exact same distributed setup.")

    def resume(self, path: str, load_capsules: bool = True) -> Self:
        """
        Sets up the launcher to resume from a checkpoint.

        This method configures the launcher to resume training from a specified
        checkpoint path. It allows control over whether to load the state of
        capsules along with the model state.

        Parameters
        ----------
        path : str
            The file path to the checkpoint from which to resume.
        load_capsules : bool, optional
            If True, loads the state of capsules along with the model state.
            If False, only loads the model state. Default is True.

        Returns
        -------
        Self
            Returns the instance of the launcher for method chaining.

        Notes
        -----
        - This method only sets up the resumption configuration. The actual
          loading of the checkpoint occurs in the `setup` method.
        - The `_resume_from` attribute is used to store the checkpoint path.
        - The `_load_capsules` attribute determines whether capsule states
          should be loaded during resumption.
        """
        self._resume_from = path
        self._load_capsules = load_capsules
        return self

    def state_dict(self) -> dict:
        """
        Returns a dictionary containing the current state of the launcher.

        This method is used to serialize the current state of the launcher,
        which can be used for saving checkpoints or transferring the state
        to another instance.

        Returns
        -------
        dict
            A dictionary containing the current epoch index.
        """
        return dict(epoch_idx=self._epoch_idx,
                    num_procs=self._num_procs,
                    num_nodes=self._num_nodes)

    def load_state_dict(self, state: Attributes) -> None:
        """
        Loads the state of the launcher from a given state dictionary.

        This method is used to deserialize the state of the launcher,
        typically when loading a checkpoint or transferring state between
        instances.

        Parameters
        ----------
        state : Attributes
            An Attributes object containing the state to be loaded into the
            launcher. It should have an 'epoch_idx' key with the corresponding
            value.

        Returns
        -------
        None
        """
        self._epoch_idx = state["epoch_idx"]
        self._num_procs = state["num_procs"]
        self._num_nodes = state["num_nodes"]
