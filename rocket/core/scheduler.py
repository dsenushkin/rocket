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


class Scheduler(Capsule):
    """
    A capsule wrapping a PyTorch learning rate scheduler for Rocket framework.

    This class integrates a PyTorch learning rate scheduler with Rocket's event
    system and accelerator. It manages the scheduler's step execution and
    ensures proper integration with distributed training.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler.LRScheduler
        The PyTorch learning rate scheduler to be wrapped.
    priority : int, optional
        The capsule's priority in the event system. Default is 1000.

    Attributes
    ----------
    _scheduler : torch.optim.lr_scheduler.LRScheduler
        The wrapped PyTorch learning rate scheduler.
    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        priority: int = 1000
    ) -> None:
        super().__init__(statefull=False, priority=priority)
        self._scheduler = scheduler

    def setup(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.SETUP` event.

        Sets up the scheduler by registering it with the accelerator. If the
        scheduler is already registered, it retrieves the existing instance.
        This method ensures that the same scheduler is not registered twice.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Raises
        ------
        RuntimeError
            If the same scheduler has been registered twice.

        Returns
        -------
        None
        """
        Capsule.setup(self, attrs=attrs)
        # safely register the scheduler, if it already exists
        # in the accelerator, just return it
        # registering the same scheduler twice is prohibited
        registered = False
        for sched in self._accelerator._schedulers:
            # search and skip non-matches
            if self._scheduler is not sched.scheduler:
                continue
            # found twice, raise an exception
            if registered:
                err = f"{self.__class__.__name__}: "
                err += "same scheduler has been registered twice. "
                raise RuntimeError(err)

            # found one, return it
            registered = True
            self._scheduler = sched

        # not found, register it
        if not registered:
            self._scheduler = self._accelerator.prepare(self._scheduler)

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.LAUNCH` event.

        This method steps the learning rate scheduler if gradient computation
        is enabled (i.e., during training). If gradients are disabled (e.g.,
        during evaluation), no action is taken.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        # if training is disabled, nothing to do
        if torch.is_grad_enabled():
            self._scheduler.step()

    def destroy(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.DESTROY` event.

        This method safely removes the scheduler from the accelerator and
        performs cleanup. It iterates through the accelerator's schedulers
        to find and remove the current scheduler.

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
        for id, scheduler in enumerate(self._accelerator._schedulers):
            if scheduler is not self._scheduler:
                continue
            _id = id
            break
        if _id is not None:
            self._accelerator._schedulers.pop(_id)

        Capsule.destroy(self, attrs=attrs)
