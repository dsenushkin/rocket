from typing import List
from operator import attrgetter

from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes, Events


class Dispatcher(Capsule):
    def __init__(self, 
                 capsules: List[Capsule], 
                 accelerator: Accelerator = None,
                 priority=1000) -> None:
        super().__init__(accelerator=accelerator,
                         statefull=False,
                         priority=priority)
        self.guard(capsules)
        self._capsules = sorted(capsules, 
                                key=attrgetter("_priority"), 
                                reverse=True)
        self.accelerate(accelerator)


    def setup(self, attrs: Attributes=None):
        # capsule states are stored in self._accelerator._custom_objects
        # this is a stack of elements with state_dict() and load_state_dict()
        # in setup() all statefull capsules register their state in _custom_objects
        # in destroy() all statefull capsules pops theis state from it
        # IMPORTANT: in destroy we should pop states in reverse order
        Capsule.setup(self, attrs=attrs)
        # call hidden capsules
        for capsule in self._capsules:
            capsule.dispatch(Events.SETUP, attrs)


    def destroy(self, attrs: Attributes=None):
        # capsule states are stored in self._accelerator._custom_objects
        # this is a stack of elements with state_dict() and load_state_dict()
        # in setup() all statefull capsules register their state in _custom_objects
        # in destroy() all statefull capsules pops theis state from it
        # IMPORTANT: in destroy we should pop states in reverse order
        for capsule in reversed(self._capsules):
            capsule.dispatch(Events.DESTROY, attrs)
        # call default behavior
        Capsule.destroy(self, attrs=attrs) 

    # set, launch, reset runs in direct order always
    def set(self, attrs: Attributes=None):
        Capsule.set(self, attrs=attrs)
        # call hidden capsules
        for capsule in self._capsules:
            capsule.dispatch(Events.SET, attrs)


    def reset(self, attrs: Attributes=None):
        Capsule.reset(self, attrs=attrs)
        for capsule in self._capsules:
            capsule.dispatch(Events.RESET, attrs)
        # call default behavior


    def launch(self, attrs: Attributes=None):
        # call default behavior
        Capsule.launch(self, attrs=attrs)
        # call hidden capsules
        for capsule in self._capsules:
            capsule.dispatch(Events.LAUNCH, attrs)
    

    def accelerate(self, accelerator: Accelerator):
        # call default behavior
        Capsule.accelerate(self, accelerator)
        # call hidden capsules
        for capsule in self._capsules:
            capsule.accelerate(accelerator)


    def guard(self, capsules: List[Capsule]):
        for capsule in capsules:
            if not isinstance(capsule, Capsule):
                err = f"{self.__class__.__name__} got invalid capsule."
                raise ValueError(err)


    def __repr__(self) -> str:
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
