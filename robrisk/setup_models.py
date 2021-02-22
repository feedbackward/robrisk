'''Setup: models.'''

## External modules.
import numpy as np

## Internal modules.
from mml.models import init_range
from mml.models.linreg import LinearRegression, LinearRegression_Multi


###############################################################################


## The main parser function, returning model instances.

def get_model(name, paras_init=None, rg=None, **kwargs):

    ## Initializer preparation.
    if paras_init is None:
        try:
            ## If given w_star, use it (w/ noise).
            w_init = np.copy(kwargs["w_star"])
            w_init += rg.uniform(low=-init_range,
                                 high=init_range,
                                 size=w_init.shape)
            paras_init = {}
            paras_init["w"] = w_init
        except KeyError:
            ## If no w_star given, do nothing special.
            pass
    else:
        paras_init = paras_init

    ## Instantiate the desired model.
    if name == "linreg_multi":
        model_out = LinearRegression_Multi(num_features=kwargs["num_features"],
                                           num_outputs=kwargs["num_classes"],
                                           paras_init=paras_init,
                                           rg=rg)
    elif name == "linreg":
        model_out = LinearRegression(num_features=kwargs["num_features"],
                                     paras_init=paras_init, rg=rg)
    else:
        raise ValueError("Please pass a valid model name.")
    
    ## Whenever needed, initialize the CVaR shift para.
    if kwargs["use_cvar"] and "v" not in model_out.paras:
        model_out.paras["v"] = rg.uniform(low=0.0, high=init_range,
                                          size=(1,1))

    ## Finally, return the completely initialized model.
    return model_out
    

###############################################################################
