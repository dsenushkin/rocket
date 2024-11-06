import os
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

from typing import List
from accelerate.tracking import GeneralTracker

from rocket.core.capsule import Capsule, Attributes


class Tracker(Capsule):
    """
    A capsule for tracking and logging experiment data in Rocket framework.

    This class integrates various tracking backends (e.g., TensorBoard) with
    Rocket's event system and accelerator. It manages tracker initialization,
    logging of scalar values and images, and handles data buffering for logs.

    Parameters
    ----------
    backend : str, optional
        The name of the tracking backend to use. Default is "tensorboard".
    tag : str, optional
        A tag for the experiment. Default is "exp_0".
    config : dict, optional
        Configuration dictionary for the tracker. Default is None.
    priority : int, optional
        The capsule's priority in the event system. Default is 200.

    Attributes
    ----------
    _backend : str
        The name of the tracking backend being used.
    _tracker : object
        The actual tracker object, initialized during setup.
    _tag : str
        The tag used for the experiment.
    _config : dict
        Configuration dictionary for the tracker.
    """

    def __init__(
        self,
        backend: str = "tensorboard",
        config: dict = None,
        priority: int = 200
    ) -> None:
        super().__init__(priority=priority)
        self._backend = backend
        self._tracker = None
        self._config = config or None

    def setup(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.SETUP` event.

        Initializes the tracker by getting or creating the specified backend
        tracker. If the tracker is not initialized, it attempts to create it.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Raises
        ------
        RuntimeError
            If the tracker cannot be created.

        Returns
        -------
        None
        """
        Capsule.setup(self, attrs=attrs)
        if isinstance(self._backend, GeneralTracker):
            self._tracker = self._backend
        else:
            self._tracker = self._accelerator.get_tracker(self._backend)

        if type(self._tracker) == GeneralTracker:   # noqa E721
            self._logger.warn(
                f"Accelerator has not initialized {self._backend}. "
                "Trying to create it..."
            )

            try:
                project_name = os.path.basename(self._accelerator.project_dir)
                self._accelerator.log_with.append(self._backend)
                self._accelerator.init_trackers('', self._config)
            except Exception as e:
                raise RuntimeError(
                    f"{self.__class__.__name__} can't create tracker: {e}"
                )

        self._tracker = self._accelerator.get_tracker(self._backend)

    def set(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.SET` event.

        Initializes the tracker attributes in the global buffer with empty
        lists for scalars and images.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        Capsule.set(self, attrs=attrs)
        attrs.tracker = Attributes(scalars=[], images=[])

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.LAUNCH` event.

        Logs the accumulated images and scalars from the tracker attributes
        in the global buffer, then resets the tracker attributes.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        Capsule.launch(self, attrs=attrs)
        # The tracker expects the global buffer and its fields to be set
        if attrs is None or attrs.tracker is None:
            return

        # if the buffer is empty, there's nothing to log
        if not attrs.tracker.images and not attrs.tracker.scalars:
            return

        self.log(attrs.tracker.images, attrs.tracker.scalars)
        attrs.tracker = Attributes(scalars=[], images=[])

    def reset(self, attrs: Attributes = None) -> None:
        """
        Handler for the :class:`Events.RESET` event.

        Logs any remaining images and scalars from the tracker attributes
        in the global buffer, then removes the tracker attributes.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        Capsule.reset(self, attrs=attrs)

        if attrs is None or attrs.tracker is None:
            return

        # if the buffer is empty, there's nothing to log
        if not attrs.tracker.images and not attrs.tracker.scalars:
            return

        self.log(attrs.tracker.images, attrs.tracker.scalars)
        del attrs.tracker

    def destroy(self, attrs: Attributes = None) -> None:
        """
        Handler for the :class:`Events.DESTROY` event.

        Removes the tracker instance and calls the parent class's
        destroy method.

        Parameters
        ----------
        attrs : Attributes | None, optional
            The global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        del self._tracker
        Capsule.destroy(self, attrs=attrs)

    def log(
        self,
        images: List[Attributes] | None,
        scalars: List[Attributes] | None
    ) -> None:
        """
        Log images and scalars to the tracking backend.

        Logs images and scalars to the configured tracking backend if the
        current process is the main process. Handles image and scalar logging
        separately, with error handling for each.

        Parameters
        ----------
        images : list
            List of image objects to log. Each should have 'data' and 'step'
            attributes.
        scalars : list
            List of scalar objects to log. Each should have 'data' and 'step'
            attributes.

        Raises
        ------
        RuntimeError
            If there's an error while logging images or scalars.

        Notes
        -----
        - Logging only occurs on the main process to avoid duplicates in
          distributed training.
        - Debug messages are logged upon successful logging of images and
          scalars.
        """
        # if images are not empty
        if images and self._accelerator.is_main_process:
            try:
                for image in images:
                    self._tracker.log_images(image.data, step=image.step)
                    self._logger.debug(
                        f"Successfully logged images to {self._backend}"
                    )
            except Exception as e:
                raise RuntimeError(f"Can't log images: {e}")

        # if scalars are not empty
        if scalars and self._accelerator.is_main_process:
            try:
                for scalar in scalars:
                    self._tracker.log(scalar.data, step=scalar.step)
                    self._logger.debug(
                        f"Successfully logged scalars to {self._backend}"
                    )
            except Exception as e:
                raise RuntimeError(f"Can't log scalars: {e}")
