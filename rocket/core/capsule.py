import logging
from enum import Enum
from adict import adict

from accelerate import Accelerator
from accelerate.logging import get_logger




Attributes =  adict


class Events(Enum):
    SETUP = "setup"
    DESTROY = "destroy"
    SET = "set"
    RESET = "reset"
    LAUNCH = "launch"



class Capsule:
    def __init__(self,
                 statefull: bool=False,
                 accelerator: Accelerator=None,
                 logger: logging.Logger=None,
                 priority: int = 1000) -> None:
        super().__init__()
        self._priority = priority
        self._statefull = statefull
        self._accelerator = accelerator or None
        self._logger = logger or get_logger(self.__module__)


    def setup(self, attrs: Attributes=None):
        # validate accelerator
        self.check_accelerator()
        # register state if statefull
        if self._statefull:
            # capsule states are stored in self._accelerator._custom_objects
            # this is a stack of elements with state_dict() and load_state_dict()
            # in setup() all statefull capsules register their state in _custom_objects
            # in destroy() all statefull capsules pops theis state from it
            # IMPORTANT: in destroy we should pop states in reverse order
            self._accelerator.register_for_checkpointing(self)

        # default setup log
        message = f"{self.__class__.__name__} initialized."
        # base logging level is INFO
        self._logger.debug(message)


    def destroy(self, attrs: Attributes=None):
        # unregister state if statefull
        if self._statefull:
            # safe call for unregister
            obj = self._accelerator._custom_objects.pop()
            # poped object must be the same as we register
            if obj is not self:
                err = f"{self.__class__.__name__}: "
                err += "illegal destroy request. "
                err += f"{obj.__class__.__name__}"
                raise RuntimeError(err)

        # default destroy log
        message = f"{self.__class__.__name__} destroyed."
        # base logging level is INFO
        self._logger.debug(message)


    def launch(self, attrs: Attributes=None):
        # default log
        message = f"{self.__class__.__name__}.launch() called "
        message += f"with attributes {attrs}"
        # default behavior is used for debugging
        self._logger.debug(message)



    def set(self, attrs: Attributes=None):
        # default log
        message = f"{self.__class__.__name__}.set() called "
        message += f"with attributes {attrs}"
        # default behavior is used for debugging
        self._logger.debug(message)


    def reset(self, attrs: Attributes=None):
        # default log
        message = f"{self.__class__.__name__}.reset() called "
        message += f"with attributes {attrs}"
        # default behavior is used for debugging
        self._logger.debug(message)


    def dispatch(self, event: Events, attrs: Attributes=None):
        return getattr(self, event.value)(attrs)


    def accelerate(self, accelerator: Accelerator):
        self._accelerator = accelerator
    

    def set_logger(self, logger: logging.Logger):
        self._logger = logger


    def check_accelerator(self):
        if self._accelerator is None:
            err = f"{self.__class__.__name__}: accelerator is not defined. "
            err += "Please, specify it in __init__ function "
            err += "or set it via .accelerate(accelerator) method."
            raise RuntimeError(err)
    
    def state_dict():
        pass

    def load_state_dict():
        pass


    def __repr__(self) -> str:
        tabs = " " * 4
        def reformat(value):
            return str(value).replace("\n", f"\n{tabs*2}")
        
        attrs = f"\n{tabs}".join(
            f"{key}={reformat(value)}" 
            for key, value in self.__dict__.items()
        )
        return f"{self.__class__.__name__}(\n{tabs}{attrs}\n)"