'''Setup: algorithms.'''

## External modules.
import numpy as np

## Internal modules.
from mml.algos.gd import GD_ERM


###############################################################################


## Simple parser for algorithm objects.

def get_algo(name, model, loss, **kwargs):

    if name == "SGD":
        return GD_ERM(step_coef=kwargs["step_size"],
                      model=model,
                      loss=loss)
    else:
        return None
                      

###############################################################################
