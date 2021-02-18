'''Setup: algorithms.'''

## External modules.
import numpy as np

## Internal modules.
from mml.algos.gd import GD_ERM
from mml.algos.rgd import RGD_Mest
from mml.utils.mest import est_loc_fixedpt, inf_gudermann, est_scale_chi_fixedpt, chi_geman_quad


###############################################################################


## Detailed setup for algorithms.

_inf_fn = inf_gudermann # influence function.
_est_loc = lambda X, s, thres, iters: est_loc_fixedpt(X=X, s=s,
                                                      inf_fn=_inf_fn,
                                                      thres=thres,
                                                      iters=iters)
_chi_fn = chi_geman_quad # chi function.
_est_scale = lambda X: est_scale_chi_fixedpt(X=X, chi_fn=_chi_fn)


## Simple parser for algorithm objects.

def get_algo(name, model, loss, **kwargs):

    if name == "SGD":
        return GD_ERM(step_coef=kwargs["step_size"],
                      model=model,
                      loss=loss)
    elif name == "RGD-M":
        return RGD_Mest(est_loc=_est_loc,
                        est_scale=_est_scale,
                        delta=0.01,
                        mest_thres=1e-03,
                        mest_iters=50,
                        step_coef=kwargs["step_size"],
                        model=model,
                        loss=loss)
    else:
        return None
                      

###############################################################################
