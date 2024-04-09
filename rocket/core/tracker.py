
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
                 priority: int = 1000) -> None:
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
            
        if isinstance(self._tracker, GeneralTracker):

            wrn = f"{self.__class__.__name__}: "
            wrn += f"accelerator does not contain {self._backend}."
            wrn += " Trying to create it..."
            self._logger.warn(wrn)

            try:
                self._accelerator.log_with.append(self._backend)
                self._accelerator.init_trackers(self._project, self._project_config)
            except Exception as e:
                err = f"{self.__class__.__name__} failed to create tracker: {e}"
                raise RuntimeError(err)
            
        self._tracker = self._accelerator.get_tracker(self._backend)
    

    def launch(self, attrs: Attributes = None):
        Capsule.launch(self, attrs=attrs)
        
        # tracker expects attrs.tracker to be defined
        if attrs is None or attrs.tracker is None:
            return
        
        if attrs.tracker.images is not None and hasattr(self._tracker, "log_images"):
            self._tracker.log_images(attrs.tracker.images, step=self._iter_idx)
            self._logger.debug("Successfully logged images to TensorBoard")
        
        self._iter_idx += 1
    

    def destroy(self, attrs: Attributes = None):
        del self._tracker
        Capsule.destroy(self, attrs=attrs)
        

    def state_dict(self):
        return Attributes(iter_idx=self._iter_idx)
    
    def load_state_dict(self, state):
        self._iter_idx = state.iter_idx

