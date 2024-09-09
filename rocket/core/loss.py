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


class Loss(Capsule):
    """
    A capsule for managing and calculating loss during training.

    This class extends the Capsule class and provides functionality for
    computing loss, handling gradient accumulation, and updating trackers.
    It is designed to work with distributed training setups and supports
    gradient accumulation.

    Parameters
    ----------
    objective : torch.nn.Module
        The loss function to be used for calculating the loss.
    tag : str, optional
        A tag for identifying the loss in logging and tracking.
        Default is "train_loss".
    priority : int, optional
        The priority of this capsule in the pipeline. Default is 1100.

    Attributes
    ----------
    _objective : torch.nn.Module
        The loss function used for calculations.
    _value : float
        The accumulated loss value.
    _tag : str
        The tag used for logging and tracking the loss.
    _step : int
        The current step count for tracking purposes.
    """

    def __init__(
        self,
        objective: torch.nn.Module,
        tag: str = "train_loss",
        priority: int = 1100          # priority higher than optimizer
    ) -> None:
        super().__init__(statefull=True,
                         priority=priority)
        self._objective = objective
        self._value = 0.0
        self._tag = tag
        self._step = 0

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.LAUNCH` event.

        This method calculates the loss for the current batch, aggregates it
        across processes, and handles gradient accumulation and synchronization.
        It also updates trackers and manages the loss state.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Attributes containing the current batch and other relevant
            information. If None or if attrs.batch is None, the method returns
            without action.

        Returns
        -------
        None
        """
        # Loss needs a non-empty buffer and batch to work
        if attrs is None or attrs.batch is None:
            return

        # Loss is not calculated if gradient is disabled
        if not torch.is_grad_enabled():
            return

        # calculate loss
        loss = self._objective(attrs.batch)

        # aggregate from other processes
        gathered_loss = self._accelerator.gather(loss).mean()
        # account for the accumulation multiplier
        self._value += gathered_loss.item() / \
            self._accelerator.gradient_accumulation_steps

        # accumulation is complete, gradients are synchronized
        if self._accelerator.sync_gradients:
            # send value to the tracker
            if attrs.tracker is not None:
                state = Attributes(
                    step=self._step,
                    data={self._tag: self._value}
                )
                # attrs.tracker.scalars.update({self._tag: self._value})
                attrs.tracker.scalars.append(state)

            if attrs.looper is not None:
                attrs.looper.state.loss = self._value

            # reset buffer for tracker
            self._value = 0.0
            self._step += 1

        # calculate gradient
        self._accelerator.backward(loss)

    def state_dict(self) -> dict:
        """
        Returns a dictionary containing the current state of the loss module.

        This method serializes the current state of the loss module for
        saving checkpoints or transferring state to another instance.

        Returns
        -------
        dict
            A dictionary with the current value and step of the loss module.
        """
        return dict(value=self._value, step=self._step)

    def load_state_dict(self, state: dict):
        """
        Loads the state of the loss module from a given state dictionary.

        This method deserializes the state of the loss module, typically
        when loading a checkpoint or transferring state between instances.

        Parameters
        ----------
        state : dict
            A dictionary with the state to load into the loss module.
            It should contain 'value' and 'step' keys with corresponding
            values.
        """
        self._value = state["value"]
        self._step = state["step"]
