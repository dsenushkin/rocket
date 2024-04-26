import os
from accelerate import Accelerator

from rocket.core.capsule import Capsule, Attributes


class Checkpointer(Capsule):
    def __init__(self,
                 output_dir: str,
                 resume_from: str = None,
                 resume_capsules: bool = True,
                 save_every: int = None,
                 overwrite: bool = True,
                 statefull: bool = True,
                 accelerator: Accelerator = None, 
                 priority: int = 100) -> None:
        super().__init__(accelerator=accelerator, 
                         statefull=statefull,
                         priority=priority)
        self._save_every = save_every or -1
        self._output_dir = output_dir
        self._resume_from = resume_from or None
        self._resume_capsules = resume_capsules
        self._overwrite = overwrite
        self._iter_idx = 0
    
    def setup(self, attrs: Attributes=None):
        # output log for humans
        Capsule.setup(self, attrs=attrs)
        # load checkpoint if needed
        if self._resume_from is not None:
            if not self._resume_capsules:
                # temporarily remove custom objects from accelerator
                custom_objects = self._accelerator._custom_objects
                self._accelerator._custom_objects = []
                try:
                    # supress RuntimeError caused by difference in 
                    # detected custom_checkpoint_* files and 
                    # custom objects registered in register_for_checkpointing
                    self._accelerator.load_state(self._resume_from)
                except RuntimeError:
                    # set them back for saving states in future
                    self._accelerator._custom_objects = custom_objects
            else:
                # resume full state with capsules
                self._accelerator.load_state(self._resume_from)


    def launch(self, attrs: Attributes=None):
        # debug log for humans
        Capsule.launch(self, attrs=attrs)

        if not self._accelerator.is_main_process:
            return

        # nothing to do
        if self._save_every < 0:
            return
        
        # everything that passed in accelerate.prepare 
        # or accelerate.register_for_chekpointing will be saved
        if (self._iter_idx + 1) % self._save_every == 0:
            # self._accelerator.wait_for_everyone()

            output_dir = os.path.join(self._output_dir, str(self._iter_idx))
            if not self._overwrite and os.path.exists(output_dir):
                err = f"{self.__class__.__name__}: overwrite is set to False. "
                err += f"{output_dir}"
                raise RuntimeError(err)

            self._accelerator.save_state(output_dir=output_dir)
            self._logger.info(f"{self.__class__.__name__}: saved {output_dir}")
        self._iter_idx += 1
        

    def state_dict(self):
        # +1 since launch saves previous index
        return Attributes(iter_idx=self._iter_idx + 1)
    
    
    def load_state_dict(self, state):
        self._iter_idx = state.iter_idx