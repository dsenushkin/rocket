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

from rocket.core.capsule import Capsule, Attributes


class Optimizer(Capsule):
    """
    A capsule wrapping a PyTorch optimizer for use in the Rocket framework.

    This class integrates a PyTorch optimizer with Rocket's event system and
    accelerator. It manages optimizer step execution, gradient zeroing, and
    logging of optimizer states.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The PyTorch optimizer to be wrapped.
    tag : str, optional
        A tag for logging purposes. Default is "opt".
    priority : int, optional
        The capsule's priority in the event system. Default is 1000.

    Attributes
    ----------
    _optimizer : torch.optim.Optimizer
        The wrapped PyTorch optimizer.
    _tag : str
        The tag used for logging.
    _iter_idx : int
        Counter for optimization steps taken.

    Notes
    -----
    This capsule works with accelerator and Rocket's event system, ensuring
    proper integration with distributed training and mixed precision.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        tag: str = "opt",
        priority: int = 1000
    ) -> None:
        super().__init__(statefull=False,
                         priority=priority)
        self._optimizer = optimizer
        self._tag = tag
        self._iter_idx = 0

    def setup(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.SETUP` event.

        Sets up the optimizer by registering it with the accelerator. If the
        optimizer is already registered, it retrieves the existing instance.
        This method ensures that the same optimizer is not registered twice.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Raises
        ------
        RuntimeError
            If the same optimizer has been registered twice.

        Returns
        -------
        None
        """
        Capsule.setup(self, attrs=attrs)

        registered = False
        # safely register the optimizer, if it already exists
        # in the accelerator, just return it
        # registering the same optimizer twice is prohibited
        for optim in self._accelerator._optimizers:
            # search and skip non-matches
            if self._optimizer is not optim.optimizer:
                continue
            # found twice, raise an exception
            if registered:
                raise RuntimeError(
                    f"{self.__class__.__name__}: "
                    "same optimizer has been registered twice."
                )

            # found one, return it
            registered = True
            self._optimizer = optim

        # not found, register it
        if not registered:
            self._optimizer = self._accelerator.prepare(self._optimizer)

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.LAUNCH` event.

        Performs the optimization step if gradients are enabled, and updates
        the learning rate tracking information.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        # The wrapped optimizer understands when to skip steps
        if torch.is_grad_enabled():
            self._optimizer.step()
            self._optimizer.zero_grad()

        # Post optimizer state for the tracker when taking a step
        if self._accelerator.sync_gradients:
            state = Attributes(
                step=self._iter_idx,
                data={
                    f"{self._tag}.lr.{idx}": group.get("lr")
                    for idx, group in enumerate(self._optimizer.param_groups)
                }
            )
            if attrs.tracker is not None:
                attrs.tracker.scalars.append(state)

            if attrs.looper is not None:
                attrs.looper.state.lr = list(state.data.values())

            self._iter_idx += 1

    def destroy(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the Events.DESTROY event.

        Safely removes the optimizer from the accelerator and performs cleanup.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        # safely remove from the accelerator
        _id = None
        for id, optimizer in enumerate(self._accelerator._optimizers):
            if optimizer is not self._optimizer:
                continue
            _id = id
            break

        if _id is not None:
            self._accelerator._optimizers.pop(_id)

        Capsule.destroy(self, attrs=attrs)

    def state_dict(self) -> dict:
        """
        Returns a dictionary containing the current state of the optimizer.

        This method serializes the current state of the optimizer for
        saving checkpoints or transferring state to another instance.

        Returns
        -------
        dict
            A dictionary with the current iteration index of the optimizer.
        """
        return dict(iter_idx=self._iter_idx)

    def load_state_dict(self, state: dict) -> None:
        """
        Loads the state of the optimizer from a given state dictionary.

        This method deserializes the state of the optimizer, typically
        when loading a checkpoint or transferring state between instances.

        Parameters
        ----------
        state : dict
            A dictionary with the state to load into the optimizer.
            It should contain an 'iter_idx' key with the corresponding value.
        """
        self._iter_idx = state["iter_idx"]
