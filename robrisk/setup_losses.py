'''Setup: loss functions used for training and evaluation.'''

## Internal modules.
from mml.losses.classification import Zero_One
from mml.losses.cvar import CVaR
from mml.losses.logistic import Logistic
from mml.losses.quadratic import Quadratic


###############################################################################


## A dictionary of instantiated losses.

dict_losses = {
    "logistic": Logistic(),
    "quadratic": Quadratic(),
    "zero_one": Zero_One()
}

def get_loss(name, **kwargs):
    '''
    A simple parser that returns a loss instance.
    '''
    if kwargs["use_cvar"]:
        return CVaR(loss_base=dict_losses[name],
                    alpha=kwargs["alpha"])
    else:
        return dict_losses[name]


###############################################################################
