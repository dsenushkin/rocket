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

import torch
from typing import Callable
from tqdm import tqdm
from termcolor import colored

from rocket.core.capsule import Capsule, Attributes
from rocket.core.dataset import Dataset
from rocket.core.dispatcher import Dispatcher


class Looper(Dispatcher):
    """
    A class for managing looping behavior in the Rocket framework.

    This class extends the Dispatcher class and provides functionality for
    repeating a set of operations (capsules) a specified number of times.
    It can be configured to run at specific intervals during the training
    process.

    Attributes:
    -----------
    _statefull : bool
        Whether the Looper maintains state across iterations.
    _repeats : int | None
        The number of times to repeat the loop, calculated at runtime.
    _user_defined_repeats : int | None
        The user-specified number of repetitions.
    _grad_enabled : bool
        Whether gradients are enabled during the loop execution.
    _run_every : int
        The frequency at which this Looper should run (e.g., every N epochs).
    _iter_idx : int
        The current iteration index.
    _tag : str
        A string identifier for this Looper instance.

    Parameters:
    -----------
    capsules : list[Capsule]
        A list of Capsule objects to be executed in the loop.
    tag : str, optional
        An identifier for this Looper. Defaults to "Looper".
    grad_enabled : bool, optional
        Whether to enable gradients. Defaults to True.
    repeats : int | None, optional
        The number of times to repeat the loop. If None, will be inferred.
    run_every : int, optional
        How often to run this Looper (e.g., every N epochs). Defaults to 1.
    statefull : bool, optional
        Whether this Looper maintains state. Defaults to True.
    priority : int, optional
        The priority of this Looper in the event handling queue. Defaults to
        1000.
    """

    def __init__(
        self,
        capsules: list[Capsule],
        tag: str = "Looper",
        grad_enabled: bool = True,
        repeats: int | None = None,
        run_every: int = 1,
        statefull: bool = True,
        priority: int = 1000
    ) -> None:
        super().__init__(capsules=capsules, priority=priority)
        self._statefull = statefull
        # number of repetitions used by class methods
        self._repeats = None
        # user-defined repetitions, necessary for set
        self._user_defined_repeats = repeats or None
        self._grad_enabled = grad_enabled
        self._run_every = run_every
        self._iter_idx = 0
        self._tag = tag

    def run_if_needed(method: Callable) -> Callable:
        """
        Decorator to conditionally execute a method based on the current epoch.

        This decorator checks if the current epoch is a multiple of the
        Looper's `_run_every` attribute. If so, it executes the decorated
        method; otherwise, it skips execution.

        Parameters
        ----------
        method : Callable
            The method to be decorated.

        Returns
        -------
        Callable
            A wrapper function that conditionally executes the original method.
        """
        def wrapper(self, attrs: Attributes | None = None):
            epoch = attrs.launcher.epoch_idx
            if epoch % self._run_every == 0:
                method(self, attrs=attrs)
        return wrapper

    @run_if_needed
    def set(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.SET` event.

        This method sets up the Looper for execution. It initializes the number
        of repeats, either from user-defined values or by inference. It also
        sets up the looper attributes in the global exchange buffer.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Raises
        ------
        RuntimeError
            If the number of repeats is not specified and cannot be inferred,
            potentially leading to an infinite loop.

        Returns
        -------
        None
        """
        Dispatcher.set(self, attrs=attrs)

        self._repeats = self._user_defined_repeats

        if self._repeats is None:
            self.infer_repeats()

        if self._repeats is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: infinite loops are not allowed. "
                "Please, specify number of repeats."
            )

        if attrs.looper is None:
            attrs.looper = Attributes(
                repeats=self._repeats,
                state=Attributes(),
                terminate=False,
                tag=self._tag
            )

    @run_if_needed
    def reset(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.RESET` event.

        This method resets the Looper to its initial state. It clears the
        number of repeats and removes the looper attributes from the global
        exchange buffer.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None
        """
        Dispatcher.reset(self, attrs=attrs)
        self._repeats = None
        del attrs.looper

    @run_if_needed
    def launch(self, attrs: Attributes | None) -> None:
        """
        Handles the :code:`Events.LAUNCH` event.

        This method executes the main loop of the Looper, iterating for a
        specified number of repeats or until terminated. It manages the
        progress bar, gradient context, and triggers the launch event for
        child capsules.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None
        """
        epoch_idx = attrs.launcher.epoch_idx

        desc = f"{colored(self._tag, 'green')} "
        desc += f"epoch={epoch_idx}, "
        desc += f"grad={self._grad_enabled}"

        status_bar = tqdm(range(self._repeats),
                          initial=0,
                          desc=desc,
                          # show progress only on the local host
                          disable=not self._accelerator.is_local_main_process)

        for _ in range(self._repeats):
            # clear batch after iteration
            attrs.batch = None
            # set gradient context
            with torch.set_grad_enabled(self._grad_enabled):
                # trigger event
                Dispatcher.launch(self, attrs)
            # shortcut to exit the loop
            # other capsules can set terminate
            if attrs.looper.terminate:
                break
            # update status bar
            status_bar.set_postfix(attrs.looper.state)
            status_bar.update(1)

        self._iter_idx = 0
        self._repeats = -1

    def state_dict(self) -> dict:
        """
        Returns a dictionary containing the state of the Looper.

        This method serializes the current state of the Looper, which can be
        used for saving checkpoints or transferring the state to another
        instance.

        Returns
        -------
        dict
            A dictionary containing the current iteration index.
        """
        return dict(iter_idx=self._iter_idx)

    def load_state_dict(self, state: dict) -> None:
        """
        Loads the state into the Looper from a dictionary.

        This method deserializes the state of the Looper, typically when
        loading a checkpoint or transferring state between instances.

        Parameters
        ----------
        state : dict
            A dictionary containing the state to be loaded into the Looper.
            It should have an 'iter_idx' key with the corresponding value.

        Returns
        -------
        None
        """
        self._iter_idx = state.get("iter_idx")

    def guard(self, capsules: list[Capsule]):
        """
        Checks if the given capsules are valid for this Looper.

        This method ensures that no internal Looper instances are present
        among the capsules, as nested Loopers are not allowed.

        Parameters
        ----------
        capsules : list[Capsule]
            A list of Capsule instances to be checked.

        Raises
        ------
        RuntimeError
            If any of the capsules is an instance of Looper.

        Returns
        -------
        None
        """
        super().guard(capsules)
        for capsule in capsules:
            if isinstance(capsule, Looper):
                raise RuntimeError(
                    f"{self.__class__.__name__}: "
                    "internal loopers are not allowed."
                )

    def infer_repeats(self):
        """
        Infers the number of repeats based on the total items in Dataset
        capsules.

        This method calculates the total number of items across all Dataset
        capsules and sets it as the number of repeats for the Looper. If no
        Dataset capsules are found, the repeats remain unchanged.

        Returns
        -------
        None

        Notes
        -----
        - This method assumes that Dataset capsules have a '_total' attribute.
        - The inferred number of repeats is logged using the class logger.
        """
        repeats = 0
        for capsule in self._capsules:
            # check dataset descendants, they have a _total field
            if isinstance(capsule, Dataset):
                repeats += capsule._total

        if repeats:
            self._repeats = repeats

        self._logger.info(
            f"{self.__class__.__name__} infered {self._repeats} repeats."
        )
