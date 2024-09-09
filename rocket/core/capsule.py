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

import logging
from enum import Enum
from adict import adict

from accelerate import Accelerator
from accelerate.logging import get_logger


Attributes = adict

"""
Attributes is the main tool for capsule interaction.

It serves as a dynamic data exchange buffer. It has a dictionary
structure with dot access to fields.

Important
---------
It does not raise exceptions when accessing non-existent fields.
Instead, it returns None.
"""


class Events(Enum):
    """
    Enum class for events that a capsule can respond to.

    This enum defines the lifecycle events of a capsule, allowing for
    structured interaction and control flow within the rocket framework.

    Attributes
    ----------
    SETUP : str
        Event for capsule initialization. Triggered when the capsule
        is being set up.
    DESTROY : str
        Event for capsule destruction. Triggered when the capsule is
        being torn down.
    SET : str
        Event for setting capsule parameters. Used to configure the
        capsule before launch.
    RESET : str
        Event for resetting capsule parameters. Used to return the
        capsule to its initial state.
    LAUNCH : str
        Event for launching the main functionality of the capsule.
        Triggers the capsule's primary action.
    """

    SETUP = "setup"    # Initialization event
    DESTROY = "destroy"  # Cleanup event
    SET = "set"        # Configuration event
    RESET = "reset"    # State reset event
    LAUNCH = "launch"  # Main functionality event


class Capsule:
    """
    Base class for all capsules in the rocket framework.

    A Capsule is a modular component that responds to various lifecycle
    events. It provides a standardized interface for initialization,
    execution, and cleanup of different parts of a machine learning pipeline.
    Capsules encapsulate lazy logic that can be executed via a launcher,
    allowing for efficient and flexible pipeline construction.

    Capsule provides event handler methods (setup, destroy, launch, set, reset)
    that follow the same syntax:

    .. code-block:: python

        def handler_name(self, attrs: Attributes | None = None) -> None:
            # Handler implementation

    Attributes
    ----------
    _priority : int
        Priority of the capsule, used for ordering when multiple capsules are
        used.
    _statefull : bool
        Flag indicating whether the capsule maintains state that should be
        checkpointed.
    _accelerator : Accelerator | None
        The accelerator object used for distributed training and mixed
        precision.
    _logger : logging.Logger
        Logger object for this capsule.
    """

    def __init__(
        self,
        statefull: bool = False,
        logger: logging.Logger | None = None,
        priority: int = 1000
    ) -> None:
        super().__init__()
        self._priority = priority
        self._statefull = statefull
        self._accelerator = None
        self._logger = logger or get_logger(self.__module__)

    def setup(self, attrs: Attributes | None = None) -> None:
        """
        Default handler for the :class:`Events.SETUP` event.

        This method initializes the capsule by checking the accelerator,
        registering for checkpointing if the capsule is stateful, and logging
        the initialization.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        self.check_accelerator()

        if self._statefull:
            # Stateful capsules are registered as arbitrary objects
            # with state using the accelerator. It knows how to save them.
            # It stores them in self._accelerator._custom_objects.
            self._accelerator.register_for_checkpointing(self)

        self._logger.debug(f"{self.__class__.__name__} initialized.")

    def destroy(self, attrs: Attributes | None = None) -> None:
        """
        Default handler for the :class:`Events.DESTROY` event.

        This method cleans up the capsule by removing it from the
        accelerator's custom objects if it's stateful. It also logs the
        capsule's destruction.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Raises
        ------
        RuntimeError
            If an attempt is made to remove an object that wasn't registered.

        Returns
        -------
        None
        """
        if self._statefull:
            obj = self._accelerator._custom_objects.pop()

            if obj is not self:
                # Attempt to remove an object that wasn't registered.
                raise RuntimeError(
                    f"{self.__class__.__name__}: Illegal destroy request. "
                    f"Attempted to remove {obj.__class__.__name__}, "
                    f"but expected {self.__class__.__name__}."
                )

        self._logger.debug(f"{self.__class__.__name__} destroyed.")

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Default handler for the :class:`Events.LAUNCH` event.

        This method is intended to be overridden by subclasses to implement
        the main functionality of the capsule. In the base implementation,
        it does nothing.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        return

    def set(self, attrs: Attributes | None = None) -> None:
        """
        Default handler for the :class:`Events.SET` event.

        This method is intended to be overridden by subclasses to set up
        the capsule's parameters before launching. In the base implementation,
        it does nothing.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        return

    def reset(self, attrs: Attributes | None = None) -> None:
        """
        Default handler for the :class:`Events.RESET` event.

        This method is intended to be overridden by subclasses to reset
        the capsule's state after launching. In the base implementation,
        it does nothing.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        return

    def dispatch(self, event: Events,
                 attrs: Attributes | None = None) -> None:
        """
        Dispatches the given event to the appropriate method.

        This method uses the event's value as the name of the method to call,
        passing the attrs parameter to that method.

        Parameters
        ----------
        event : Events
            The event to dispatch.
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        return getattr(self, event.value)(attrs)

    def accelerate(self, accelerator: Accelerator) -> None:
        """
        Sets the accelerator for this capsule.

        This method assigns the given accelerator to the capsule's
        _accelerator attribute, which is used for acceleration during
        the capsule's operation.

        Parameters
        ----------
        accelerator : Accelerator
            The accelerator to be set for this capsule.

        Returns
        -------
        None
        """
        self._accelerator = accelerator

    def clear(self) -> None:
        """
        Clears the accelerator for this capsule.

        This method removes the current accelerator by deleting the
        _accelerator attribute and then setting it to None.

        Returns
        -------
        None
        """
        del self._accelerator
        self._accelerator = None

    def set_logger(self, logger: logging.Logger) -> None:
        """
        Sets the logger for this capsule.

        This method assigns the given logger to the capsule's _logger
        attribute, which can be used for logging during the capsule's
        operation.

        Parameters
        ----------
        logger : logging.Logger
            The logger to be set for this capsule.

        Returns
        -------
        None
        """
        self._logger = logger

    def check_accelerator(self) -> None:
        """
        Checks if an accelerator is defined for this capsule.

        This method verifies if the _accelerator attribute is set. If it's not,
        it raises a RuntimeError with a descriptive message.

        Raises
        ------
        RuntimeError
            The accelerator is not defined.

        Returns
        -------
        None
        """
        if self._accelerator is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: accelerator is not defined. "
                "Please, specify it in __init__ function "
                "or set it via .accelerate(accelerator) method."
            )

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the capsule.

        This method returns a dictionary representing the current state of the
        capsule. For stateful capsules, it should be implemented by subclasses
        to include all necessary information to reconstruct the capsule's
        state.

        Returns
        -------
        dict
            A dictionary containing the capsule's current state.
            For non-stateful capsules, an empty dictionary is returned.

        Raises
        ------
        NotImplementedError
            If the capsule is stateful and this method is not implemented by a
            subclass.

        Notes
        -----
        - Stateful capsules must override this method.
        - Non-stateful capsules will return an empty dictionary.

        Examples
        --------
        For a stateful subclass:

        .. code-block:: python

            def state_dict(self):
                return {
                    'attribute1': self.attribute1,
                    'attribute2': self.attribute2.state_dict(),
                    # ... other attributes ...
                }
        """
        if not self._statefull:
            return {}
        else:
            raise NotImplementedError(
                "state_dict() must be implemented by stateful subclasses"
            )

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the state dictionary into the capsule.

        This method should be implemented by subclasses to update the
        capsule's internal state based on the provided state dictionary.
        It is typically used in conjunction with the `state_dict()` method
        for saving and loading the capsule's state.

        Parameters
        ----------
        state_dict : Attributes
            An Attributes object containing the state to be loaded into
            the capsule.

        Returns
        -------
        None

        Notes
        -----
        This is an abstract method that should be overridden by subclasses.
        The default implementation raises a :exc:`NotImplementedError`.

        Examples
        --------
        .. code-block:: python

            def load_state_dict(self, state_dict):
                self.attribute1 = state_dict['attribute1']
                self.attribute2.load_state_dict(
                    state_dict['attribute2']
                )
                # ... load other attributes ...
        """
        if not self._statefull:
            return
        else:
            raise NotImplementedError(
                "load_state_dict() must be implemented by subclasses"
            )

    def __repr__(self) -> str:
        """
        Returns a string representation of the capsule.

        This method creates a formatted string representation of the capsule,
        including all its attributes and their values.

        Returns
        -------
        str
            A string representation of the capsule.
        """
        tabs = " " * 4

        def reformat(value):
            return str(value).replace("\n", f"\n{tabs*2}")

        attrs = f"\n{tabs}".join(
            f"{key}={reformat(value)}"
            for key, value in self.__dict__.items()
        )
        return f"{self.__class__.__name__}(\n{tabs}{attrs}\n)"
