
import torch
from typing import Mapping
from logging import Logger
from accelerate import Accelerator
from accelerate.tracking import GeneralTracker

from rocket.core.capsule import Capsule, Attributes


class Tracker(Capsule):
    def __init__(self,
                 backend: str = "tensorboard",
                 project: str = "project",
                 project_config: dict = None,
                 statefull: bool = True, 
                 accelerator: Accelerator = None, 
                 logger: Logger = None, 
                 priority: int = 200) -> None:
        super().__init__(statefull=statefull, 
                         accelerator=accelerator, 
                         logger=logger, 
                         priority=priority)
        self._backend = backend
        self._tracker = None
        self._iter_idx = 0
        self._project = project
        self._project_config = project_config or None
    
    def setup(self, attrs: Attributes = None):
        Capsule.setup(self, attrs=attrs)
        self._tracker = self._accelerator.get_tracker(self._backend)
            
        if type(self._tracker) == GeneralTracker:
            wrn = f"Accelerator has not initialized {self._backend}."
            wrn += " Trying to create it..."
            self._logger.warn(wrn)

            try:
                self._accelerator.log_with.append(self._backend)
                self._accelerator.init_trackers(self._project, self._project_config)
            except Exception as e:
                err = f"{self.__class__.__name__} failed to create tracker: {e}"
                raise RuntimeError(err)
            
        self._tracker = self._accelerator.get_tracker(self._backend)


        
    def set(self, attrs: Attributes = None):
        Capsule.set(self, attrs=attrs)
        attrs.tracker = Attributes(scalars=Attributes(),
                                   images=Attributes())


    def launch(self, attrs: Attributes = None):
        Capsule.launch(self, attrs=attrs)
        # tracker expects attrs.tracker to be defined
        if attrs is None or attrs.tracker is None:
            return
        
        # pass forward in case of validation loop or accumulated optimizer step
        if torch.is_grad_enabled() and not self._accelerator.sync_gradients:
            return
        self.log(attrs)

    
    def reset(self, attrs: Attributes = None):
        Capsule.reset(self, attrs=attrs)
        self.log(attrs=attrs)
        del attrs.tracker
    
    

    def destroy(self, attrs: Attributes = None):
        del self._tracker
        Capsule.destroy(self, attrs=attrs)
        

    def state_dict(self):
        return Attributes(iter_idx=self._iter_idx)
    
    def load_state_dict(self, state):
        self._iter_idx = state.iter_idx


    def log(self, attrs):
        if not attrs.tracker.images and not attrs.tracker.scalars:
            return
        # if images are not empty and tracker can log it
        if attrs.tracker.images and hasattr(self._tracker, "log_images"):
            if not isinstance(attrs.tracker.images, Mapping):
                wrn += f"Tracker expect dict-style images for logging,"
                wrn += f" got {type(attrs.tracker.images)}"
                self._logger.warn(wrn)

            if not torch.is_grad_enabled():
                self._tracker.log_images(attrs.tracker.images, step=self._iter_idx)
                self._logger.debug(f"Successfully logged images to {self._backend}")
            elif self._accelerator.sync_gradients:
                self._tracker.log_images(attrs.tracker.images, step=self._iter_idx)
                self._logger.debug(f"Successfully logged images to {self._backend}")

        
        # if scalars are not empty, log it. Every backend can log scalars.
        if attrs.tracker.scalars:
            if not isinstance(attrs.tracker.scalars, Mapping):
                wrn += f"Tracker expect dict-style scalars for logging,"
                wrn += f" got {type(attrs.tracker.scalars)}"
                self._logger.warn(wrn)
            
            self._tracker.log(attrs.tracker.scalars, step=self._iter_idx)
            self._logger.debug(f"Successfully logged scalars to {self._backend}")

        # reset tracker buffer when logging is done
        attrs.tracker = Attributes(scalars=Attributes(),
                                   images=Attributes())
        self._iter_idx += 1