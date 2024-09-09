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

from typing import Iterable

import torch.utils.data

from rocket.core.capsule import Capsule, Attributes
from rocket.utils.torch import torch_collate, torch_move


class Dataset(Capsule):
    """
    A capsule for handling datasets in the Rocket framework.

    This class extends the Capsule base class to provide functionality for
    working with datasets, including setup, iteration, and state management.

    Parameters
    ----------
    dataset : Iterable
        The dataset to be wrapped by this capsule.
    statefull : bool, optional
        Whether this capsule maintains state (default is True).
    accelerator : Accelerator | None, optional
        The accelerator to be used for distributed training (default is None).
    priority : int, optional
        The priority of this capsule in the execution order (default is 1000).
    **kwargs
        Additional keyword arguments to be passed to the PyTorch DataLoader.

    Attributes
    ----------
    _dataset : Iterable
        The wrapped dataset object.
    _dataloader : torch.utils.data.DataLoader | None
        The default DataLoader for the dataset.
    _active_dataloader : torch.utils.data.DataLoader | None
        The current active DataLoader, which may differ from the default in
        case of state restoration.
    _iterator : Iterator | None
        The iterator over the active DataLoader.
    _kwargs : dict
        PyTorch DataLoader arguments.
    _batch_idx : int
        The current batch index.
    _total : int
        The total number of batches in the dataset.

    Notes
    -----
    This class handles the lifecycle of a dataset within the Rocket framework,
    including initialization, iteration, state management, and cleanup. It
    supports deterministic state restoration and integrates with the
    accelerator for distributed training scenarios.

    Example
    -------
    .. code-block:: python

        from rocket.core.dataset import Dataset
        from torch.utils.data import TensorDataset
        import torch

        # Create a simple dataset
        data = torch.randn(100, 5)
        labels = torch.randint(0, 2, (100,))
        tensor_dataset = TensorDataset(data, labels)

        # Create a Dataset capsule
        dataset_capsule = Dataset(
            dataset=tensor_dataset,
            batch_size=16,
            shuffle=True
        )

        # Setup the dataset
        dataset_capsule.setup()

        # Iterate through the dataset
        attrs = Attributes()
        for _ in range(5):  # Get first 5 batches
            dataset_capsule.launch(attrs)
            batch_data, batch_labels = attrs.batch
            print(f"Batch shape: {batch_data.shape}")
            attrs.batch = None  # Clear the batch for the next iteration
    """

    def __init__(
        self,
        dataset: Iterable,
        statefull: bool = True,
        priority: int = 1000,
        **kwargs
    ):
        super().__init__(statefull=statefull,
                         priority=priority)
        # Dataset object
        self._dataset = dataset
        # Default dataloader
        self._dataloader = None
        # Current dataloader, differs from default when restoring
        # class from a state with incomplete pass through the original
        self._active_dataloader = None
        # Iterator, yields data via next(self._iterator)
        self._iterator = None

        # PyTorch DataLoader arguments
        self._kwargs = kwargs
        # Modified collate function
        self._kwargs.setdefault('collate_fn', torch_collate)

        # Indexing of total size and current iteration over data
        self._batch_idx = 0
        self._total = 0

    def setup(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.SETUP` event.

        Initializes the capsule, setting up the dataloader and registering it
        with the accelerator.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Raises
        ------
        RuntimeError
            If the same dataset is registered twice.

        Returns
        -------
        None

        Notes
        -----
        Initializes and registers the dataloader, avoiding duplicates.
        """
        Capsule.setup(self, attrs=attrs)

        registered = False

        # Check for duplicate dataloader registration
        for dataloader in self._accelerator._dataloaders:
            if self._dataset is not dataloader.dataset:
                continue

            if registered:
                # Dataset registered twice. Raise an exception.
                raise RuntimeError(
                    f"{self.__class__.__name__}: "
                    "same dataset has been registered twice."
                )

            # Found the first dataloader with this dataset, store it.
            registered = True
            self._dataloader = dataloader

        # If not registered, create and register a new dataloader
        if not registered:
            self._dataloader = torch.utils.data.DataLoader(
                self._dataset, **self._kwargs
            )
            self._dataloader = self._accelerator.prepare(
                self._dataloader, device_placement=[False]
            )

    def set(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.SET` event.

        Sets up the active dataloader, either restoring a previous state or
        initializing a new one.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None

        Notes
        -----
        Sets up the active dataloader and initializes related attributes.
        """
        Capsule.set(self, attrs=attrs)

        # Restore state if different from default and not in no_grad context
        if torch.is_grad_enabled() and self._batch_idx > 0:
            self._active_dataloader = self._accelerator.skip_first_batches(
                self._dataloader, self._batch_idx
            )
        else:
            self._active_dataloader = self._dataloader

        self._total = len(self._active_dataloader)
        self._iterator = iter(self._active_dataloader)

    def reset(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.RESET` event.

        Resets the capsule's state to its initial configuration, preparing it
        for a new data iteration cycle.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None

        Notes
        -----
        Resets batch index, total count, and clears the iterator.
        """
        Capsule.reset(self, attrs=attrs)
        self._batch_idx = 0
        self._total = 0
        self._iterator = None

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.LAUNCH` event.

        Executes the main functionality of the capsule. Processes the next
        batch of data from the iterator and updates the global data exchange
        buffer accordingly.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None

        Notes
        -----
        Processes next data batch, updates buffer, and handles iteration state.
        """
        Capsule.launch(self, attrs=attrs)

        # Do nothing if the exchange buffer is not set or already occupied.
        if attrs is None or attrs.batch is not None:
            return

        data = next(self._iterator, None)

        if data is None:
            # Iterator is empty
            attrs.batch = data

            # If there was a loop, vote for exit
            if attrs.looper is not None:
                attrs.looper.terminate = True
                return
        else:
            # Iterator is not empty
            device = self._accelerator.device

            # Move to device and put in buffer
            attrs.batch = torch_move(data, device)

            # If there was a loop, vote to continue
            if attrs.looper is not None:
                attrs.looper.terminate = False

            self._batch_idx += 1

    def destroy(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.DESTROY` event.

        Called when the capsule is being destroyed. Cleans up resources and
        removes references to prevent memory leaks.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Returns
        -------
        None

        Notes
        -----
        Clears references and removes dataloader from accelerator.
        """
        Capsule.destroy(self, attrs=attrs)

        # Clear dataloader references
        self._dataloader = None
        self._active_dataloader = None

        # Clean up the accelerator
        _id = None
        for id, dataloader in enumerate(self._accelerator._dataloaders):
            if dataloader is not self._dataloader:
                continue
            _id = id
            break

        # Remove the dataloader from the accelerator if found
        if _id is not None:
            self._accelerator._dataloaders.pop(_id)

    def state_dict(self) -> dict:
        """
        Returns a dictionary containing a whole state of the capsule.

        This method is used to serialize the current state of the capsule,
        which can be used for saving checkpoints or transferring the state
        to another instance.

        Returns
        -------
        dict
            A dictionary containing the current batch index.
        """
        return dict(batch_idx=self._batch_idx)

    def load_state_dict(self, state: dict) -> None:
        """
        Copies parameters and buffers from state into this capsule.

        This method is used to deserialize the state of the capsule,
        typically when loading a checkpoint or transferring state between
        instances.

        Parameters
        ----------
        state : dict
            A dictionary containing the state to be loaded into the capsule.
            It should have a 'batch_idx' key with the corresponding value.

        Returns
        -------
        None
        """
        self._batch_idx = state['batch_idx']
