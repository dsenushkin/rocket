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

import sys
import torch

from contextlib import contextmanager

from rocket.core.dispatcher import Dispatcher
from rocket.core.capsule import Capsule, Attributes
from rocket.utils.torch import torch_move


class Module(Dispatcher):
    """
    A class representing a module in the Rocket framework.

    This class extends the Dispatcher class and encapsulates a PyTorch module,
    providing functionality for setup, launch, and destruction within the
    Rocket pipeline.

    Attributes
    ----------
    _module : torch.nn.Module
        The PyTorch module being wrapped.

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to be wrapped.
    capsules : list[Capsule], optional
        A list of capsules to be included (e.g., losses, optimizers,
        schedulers, postprocessors).
    priority : int, optional
        The priority of the module in the event handling queue
        (default is 1000).
    """

    def __init__(
        self,
        module: torch.nn.Module,
        capsules: list[Capsule] = [],   # suppose to include
                                        # losses, optimizers,
                                        # schedulers, postprocessors
        priority: int = 1000
    ) -> None:
        super().__init__(capsules=capsules,
                         priority=priority)
        self._module = module

    def setup(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.SETUP` event.

        Sets up the module by checking the accelerator, registering the module,
        and preparing it for distributed training.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Raises
        ------
        RuntimeError
            If the same module has been registered twice.

        Returns
        -------
        None
        """
        self.check_accelerator()
        # If the module is already registered, this flag = True
        registered = False
        # Check for duplication before registration
        for model in self._accelerator._models:
            # Skip non-matches
            if self._module is not model:
                continue
            # Found two, raise an exception
            if registered:
                raise RuntimeError(
                    f"{self.__class__.__name__}: "
                    "same module has been registered twice."
                )
            # Found one, take it
            registered = True
            self._module = model

        # Nothing found, register
        if not registered:
            # Move to device manually
            self._module = torch_move(self._module, self._accelerator.device)
            # Wrap in accelerator
            self._module = self._accelerator.prepare(self._module)

        Dispatcher.setup(self, attrs)

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.LAUNCH` event.

        Executes the forward pass of the module and launches other capsules.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        # The module works with a batch from the global buffer
        if attrs is None or attrs.batch is None:
            return

        # train/eval modes. Oriented by gradient
        if torch.is_grad_enabled():
            # training mode
            self._module.train()
        else:
            # eval mode
            self._module.eval()

        # Call forward pass in its context
        with self.runner():
            attrs.batch = self._module.forward(attrs.batch)
            # Call other capsules,
            # such as losses, optimizers, schedulers
            Dispatcher.launch(self, attrs=attrs)

    def destroy(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.DESTROY` event.

        Removes the module from the accelerator's model list and destroys
        the dispatcher.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        _id = None
        # Look for matches with the current one
        for id, model in enumerate(self._accelerator._models):
            # Skip non-matches
            if model is not self._module:
                continue
            # Found
            _id = id
            break
        # Remove
        if _id is not None:
            self._accelerator._models.pop(_id)

        Dispatcher.destroy(self, attrs=attrs)

    @contextmanager
    def runner(self):
        """
        A context manager that sets up the environment for running the module.

        This method creates a context that enables automatic mixed precision
        (autocast) and gradient accumulation (accumulate) for the module. It's
        designed to be used with a 'with' statement to ensure proper setup and
        teardown of these contexts.

        The autocast context allows for automatic mixed precision during
        forward pass, which can speed up computations on supporting hardware.

        The accumulate context allows for gradient accumulation over multiple
        forward and backward passes before performing a parameter update, which
        is useful for simulating larger batch sizes.

        Yields
        ------
        None
            The method yields control back to the caller within the established
            context.

        Example
        -------
        with self.runner():
            # Perform forward pass and other operations here
            pass

        Notes
        -----
        This method relies on the accelerator object to provide the autocast
        and accumulate contexts. Make sure the accelerator is properly set up
        before using this method.
        """
        autocast_ctx = self._accelerator.autocast()
        accumulate_ctx = self._accelerator.accumulate(self._module)

        try:
            autocast_ctx.__enter__()
            accumulate_ctx.__enter__()
            yield
        finally:
            autocast_ctx.__exit__(*sys.exc_info())
            accumulate_ctx.__exit__(*sys.exc_info())
