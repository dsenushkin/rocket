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
from typing import List

from rocket.core.capsule import Capsule, Attributes
from rocket.core.dispatcher import Dispatcher
from rocket.utils.collections import apply_to_collection


def rebuild_batch(lookup_table):
    def fn(value, key, **kwargs):
        # if the key exists - modify
        return lookup_table.get(key, value)
    return fn


class Meter(Dispatcher):
    """
    A class for managing metric calculations across distributed processes.

    This class extends the Dispatcher class and provides functionality for
    collecting specific keys from batches across all processes, gathering them
    for metric calculations, and reforming the batch with the gathered values.

    Attributes:
    -----------
    _keys : List[str]
        A sorted list of keys to be collected from each batch for metric
        calculation.

    Parameters:
    -----------
    capsules : List[Capsule]
        A list of Capsule objects to be executed by this Meter.
    keys : List[str]
        A list of keys to be collected from each batch for metric calculation.
    priority : int, optional
        The priority of this Meter in the pipeline. Defaults to 1000.
    """

    def __init__(
        self,
        capsules: List[Capsule],
        keys: List,
        priority=1000
    ) -> None:
        super().__init__(capsules=capsules, priority=priority)
        self._keys = sorted(keys)

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.LAUNCH` event.

        This method collects specified keys from the batch across all processes,
        gathers them for metric calculations, and reforms the batch with the
        gathered values. It only operates when gradients are disabled.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer containing the batch information.

        Returns
        -------
        None
        """
        # does nothing if the buffer is empty
        if attrs is None or attrs.batch is None:
            return

        if torch.is_grad_enabled():
            return

        # collect values from each process
        values = list()
        for key in self._keys:
            values += [attrs.batch[key]]

        # send to the global host for metric calculation
        gathered_values = self._accelerator.gather_for_metrics(values)

        # Hashtable for quick key lookup
        lookup_table = {
            k: v for k, v in zip(self._keys, gathered_values)
        }

        # reform the batch with the required keys
        attrs.batch = apply_to_collection(
            attrs.batch, rebuild_batch(lookup_table)
        )

        Dispatcher.launch(self, attrs=attrs)


class Metric(Capsule):
    """
    A base class for implementing metrics in the Rocket framework.

    This class extends the Capsule class and provides a structure for creating
    custom metrics. It includes methods for initialization, setting up the
    metric, launching the metric calculation, and resetting the metric.

    Attributes:
    -----------
    _step : int
        The current epoch index, set during the `set` method call.

    Parameters:
    -----------
    accelerator : Accelerator | None, optional
        An accelerator object for distributed computing. Defaults to None.
    priority : int, optional
        The priority of this capsule in the pipeline. Defaults to 1000.

    Methods:
    --------
    set(attrs: Attributes | None = None)
        Sets up the metric with the current epoch index.
    launch(attrs: Attributes | None = None)
        Abstract method to be implemented by subclasses for metric calculation.
    reset(attrs: Attributes | None = None)
        Abstract method to be implemented by subclasses for resetting the
        metric.
    """

    def __init__(self, priority: int = 1000) -> None:
        super().__init__(priority=priority)

    def set(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.SET` event.

        This method sets up the metric with the current epoch index.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None
        """
        Capsule.set(self, attrs)
        self._step = attrs.launcher.epoch_idx

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.LAUNCH` event.

        This method should be implemented by subclasses to perform the actual
        metric calculation.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  "metric should implement launch()")

    def reset(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.RESET` event.

        This method should be implemented by subclasses to reset the metric
        state.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  "metric should implement reset()")
