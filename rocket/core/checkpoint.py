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

from rocket.core.capsule import Capsule, Attributes


class Checkpointer(Capsule):
    """
    A class for managing checkpoints in the Rocket framework.

    This class extends the Capsule class and provides functionality for saving
    checkpoints at specified intervals during training or other iterative
    processes.

    Attributes:
    -----------
    _save_every : int
        The frequency at which checkpoints are saved. If negative, no
        checkpoints are saved.
    _output_dir : str
        The directory where checkpoints are saved.
    _overwrite : bool
        Whether to overwrite existing checkpoints.
    _iter_idx : int
        The current iteration index.

    Parameters:
    -----------
    output_dir : str
        The directory where checkpoints will be saved.
    save_every : int | None, optional
        The frequency at which to save checkpoints. If None, defaults to -1
        (no saving).
    overwrite : bool, optional
        Whether to overwrite existing checkpoints. Defaults to True.
    statefull : bool, optional
        Whether the Checkpointer maintains state. Defaults to True.
    priority : int, optional
        The priority of this Checkpointer in the event handling queue.
        Defaults to 100.
    """

    def __init__(
        self,
        output_dir: str,
        save_every: int | None = None,
        overwrite: bool = True,
        statefull: bool = True,
        priority: int = 100
    ) -> None:
        super().__init__(statefull=statefull,
                         priority=priority)
        self._save_every = save_every or -1
        self._output_dir = output_dir
        self._overwrite = overwrite
        self._iter_idx = 0

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handles the :code:`Events.LAUNCH` event.

        This method is responsible for saving checkpoints at specified
        intervals. It only runs on the main process and checks if it's time
        to save based on the current iteration index and save frequency.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer.

        Raises
        ------
        RuntimeError
            If overwrite is False and the output directory already exists.

        Returns
        -------
        None
        """
        # Call the parent method to potentially add default functionality
        Capsule.launch(self, attrs=attrs)

        if not self._accelerator.is_main_process:
            return

        # Negative period means we don't do anything
        if self._save_every < 0:
            return

        # Save all registered objects
        if (self._iter_idx + 1) % self._save_every == 0:
            output_dir = os.path.join(self._output_dir, str(self._iter_idx))

            if not self._overwrite and os.path.exists(output_dir):
                raise RuntimeError(
                    f"{self.__class__.__name__}: Cannot overwrite existing "
                    f"directory. 'overwrite' is set to False and "
                    f"'{output_dir}' already exists."
                )

            self._accelerator.save_state(output_dir=output_dir)
            self._logger.info(f"{self.__class__.__name__}: saved {output_dir}")

        self._iter_idx += 1

    def state_dict(self) -> dict:
        """
        Returns a dictionary containing the whole state of the Checkpointer.

        This method serializes the current state of the Checkpointer, which
        can be used for saving checkpoints or transferring the state to
        another instance.

        Returns
        -------
        dict
            A dictionary containing the current iteration index, incremented
            by 1 since the launch method saves the previous index.
        """
        # +1, т.к. launch сохраняет предыдущий индекс
        return dict(iter_idx=self._iter_idx + 1)

    def load_state_dict(self, state: dict) -> None:
        """
        Copies parameters and buffers from state into this Checkpointer.

        This method deserializes the state of the Checkpointer, typically
        when loading a checkpoint or transferring state between instances.

        Parameters
        ----------
        state : dict
            A dictionary containing the state to be loaded into the
            Checkpointer. It should have an 'iter_idx' key with the
            corresponding value.

        Returns
        -------
        None
        """
        self._iter_idx = state["iter_idx"]
