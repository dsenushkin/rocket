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

from operator import attrgetter

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes, Events


class Dispatcher(Capsule):
    """
    A class that manages and coordinates multiple Capsules.

    The Dispatcher class extends the Capsule class and acts as a container and
    coordinator for multiple Capsule objects. It provides methods to set up,
    destroy, set, reset, and launch all contained Capsules, as well as manage
    their acceleration and resource clearing.

    Attributes:
    -----------
    _capsules : list[Capsule]
        A sorted list of Capsule objects managed by this Dispatcher.

    Parameters:
    -----------
    capsules : list[Capsule]
        A list of Capsule objects to be managed by this Dispatcher.
    priority : int, optional
        The priority of this Dispatcher in the event handling queue
        (default is 1000).
    """

    def __init__(
        self,
        capsules: list[Capsule],
        priority: int = 1000
    ) -> None:
        super().__init__(statefull=False,
                         priority=priority)

        self.guard(capsules)
        self._capsules = sorted(capsules,
                                key=attrgetter("_priority"),
                                reverse=True)

    def setup(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.SETUP` event.

        Initializes the dispatcher and sets up all contained capsules.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        Capsule.setup(self, attrs=attrs)

        for capsule in self._capsules:
            capsule.dispatch(Events.SETUP, attrs)

    def destroy(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.DESTROY` event.

        Destroys all contained capsules in reverse order and then
        destroys the dispatcher.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        for capsule in reversed(self._capsules):
            capsule.dispatch(Events.DESTROY, attrs)

        Capsule.destroy(self, attrs=attrs)

    def set(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.SET` event.

        Sets parameters for the dispatcher and all contained capsules.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        Capsule.set(self, attrs=attrs)

        for capsule in self._capsules:
            capsule.dispatch(Events.SET, attrs)

    def reset(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.RESET` event.

        Resets the dispatcher and all contained capsules to their
        initial state.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        Capsule.reset(self, attrs=attrs)

        for capsule in self._capsules:
            capsule.dispatch(Events.RESET, attrs)

    def launch(self, attrs: Attributes | None = None) -> None:
        """
        Handler for the :class:`Events.LAUNCH` event.

        Launches the main functionality of the dispatcher and all
        contained capsules.

        Parameters
        ----------
        attrs : Attributes | None, optional
            Global data exchange buffer. Default is None.

        Returns
        -------
        None
        """
        Capsule.launch(self, attrs=attrs)

        for capsule in self._capsules:
            capsule.dispatch(Events.LAUNCH, attrs)

    def accelerate(self, accelerator: Accelerator) -> None:
        """
        Sets the accelerator for this dispatcher and all its capsules.

        Assigns the given accelerator to the dispatcher and propagates it
        to all contained capsules.

        Parameters
        ----------
        accelerator : Accelerator
            The accelerator to be set for this dispatcher and its capsules.

        Returns
        -------
        None
        """
        Capsule.accelerate(self, accelerator)

        for capsule in self._capsules:
            capsule.accelerate(accelerator)

    def clear(self) -> None:
        """
        Clears the accelerator for this dispatcher and all its capsules.

        Removes the current accelerator from the dispatcher and propagates
        this clearing operation to all contained capsules.

        Returns
        -------
        None
        """
        Capsule.clear(self)

        for capsule in self._capsules:
            capsule.clear()

    def guard(self, capsules: list[Capsule]) -> None:
        """
        Validates that all provided capsules are Capsule instances.

        Checks each capsule in the provided list to ensure it is an
        instance of the Capsule class. Raises ValueError if invalid.

        Parameters
        ----------
        capsules : list[Capsule]
            A list of capsules to be validated.

        Raises
        ------
        ValueError
            If any provided capsule is not a Capsule instance.

        Returns
        -------
        None
        """
        for capsule in capsules:
            if not isinstance(capsule, Capsule):
                raise ValueError(
                    f"{self.__class__.__name__} got invalid capsule."
                )

    def __repr__(self) -> str:
        """
        Returns a string representation of the Dispatcher.

        This method creates a formatted string representation of the
        Dispatcher, including all its attributes and their values, except
        for the '_capsules' attribute which is handled separately.

        Returns
        -------
        str
            A string representation of the Dispatcher, including its class
            name, attributes, and a list of its capsules.
        """
        tabs = " " * 4

        def reformat(value):
            return str(value).replace("\n", f"\n{tabs*2}")

        attrs = f"\n{tabs}".join(
            f"{key}={reformat(value)}"
            for key, value in self.__dict__.items() if key != "_capsules"
        )

        caps = "\n".join(str(cap) for cap in self._capsules)
        caps = caps.replace("\n", f"\n{tabs}")

        caps = f"\n_capsules=[\n{tabs}{caps}\n]"
        caps = caps.replace("\n", f"\n{tabs}")
        attrs += caps
        return f"{self.__class__.__name__}(\n{tabs}{attrs}\n)"
