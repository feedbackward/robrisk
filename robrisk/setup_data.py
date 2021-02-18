'''Setup: preparation of data sets.'''

## External modules.
import numpy as np
import os
from pathlib import Path
from tables import open_file

## Internal modules.
from mml.models.linreg import LinearRegression
from mml.utils.linalg import onehot
from mml.utils.rgen import get_generator, get_stats


###############################################################################


## Detailed specification of data to be randomly generated.

# This directory will need to be set manually.
dir_data_toread = os.path.join(str(Path.home()),
                               "mml", "mml", "data")

_n = 500
_n_train_frac = 0.8
_n_val_frac = 0.1*0.8
_n_test = 10**5
_d = 2
_init_range = 5.0

dataset_paras = {
    "ds_lognormal": {
        "n_train": _n//2,
        "n_val": _n//2,
        "n_test": _n_test,
        "num_features": _d,
        "noise_dist": "lognormal",
        "noise_paras": {"mean": 0.0, "sigma": 1.75},
        "cov_X": np.eye(_d),
        "init_range": _init_range
    },
    "ds_normal": {
        "n_train": _n//2,
        "n_val": _n//2,
        "n_test": _n_test,
        "num_features": _d,
        "noise_dist": "normal",
        "noise_paras": {"loc": 0.0, "scale": 2.2},
        "cov_X": np.eye(_d),
        "init_range": _init_range
    },
    "ds_pareto": {
        "n_train": _n//2,
        "n_val": _n//2,
        "n_test": _n_test,
        "num_features": _d,
        "noise_dist": "pareto",
        "noise_paras": {"shape": 2.1, "scale": 3.5},
        "cov_X": np.eye(_d),
        "init_range": _init_range
    },
    "adult": {"type": "classification",
              "num_classes": 2,
              "chance_level": 0.7522, # freq of the majority class.
              "n_train_frac": _n_train_frac,
              "n_val_frac": _n_val_frac},
    "australian": {"type": "classification",
                   "num_classes": 2,
                   "chance_level": 0.5551, # freq of the majority class.
                   "n_train_frac": _n_train_frac,
                   "n_val_frac": _n_val_frac},
    "cifar10": {"type": "classification",
                "num_classes": 10,
                "chance_level": 0.1,
                "pix_h": 32,
                "pix_w": 32,
                "channels": 3,
                "n_train_frac": _n_train_frac,
                "n_val_frac": _n_val_frac},
    "cod_rna": {"type": "classification",
                "num_classes": 2,
                "chance_level": 0.6666, # freq of the majority class.
                "n_train_frac": _n_train_frac,
                "n_val_frac": _n_val_frac},
    "covtype": {"type": "classification",
                "num_classes": 7,
                "n_train_frac": _n_train_frac,
                "n_val_frac": _n_val_frac},
    "emnist_balanced": {"type": "classification",
                        "num_classes": 47,
                        "chance_level": 1/47,
                        "pix_h": 28,
                        "pix_w": 28,
                        "channels": 1,
                        "n_train_frac": _n_train_frac,
                        "n_val_frac": _n_val_frac},
    "ex_quad": {"type": "regression",
                "n_train_frac": 1.0},
    "fashion_mnist": {"type": "classification",
                      "num_classes": 10,
                      "chance_level": 0.1,
                      "pix_h": 28,
                      "pix_w": 28,
                      "channels": 1,
                      "n_train_frac": _n_train_frac,
                      "n_val_frac": _n_val_frac},
    "iris": {"type": "classification",
             "num_classes": 3,
             "chance_level": 0.3,
             "n_train_frac": _n_train_frac,
             "n_val_frac": _n_val_frac,
             "init_range": _init_range},
    "mnist": {"type": "classification",
              "num_classes": 10,
              "chance_level": 0.1,
              "pix_h": 28,
              "pix_w": 28,
              "channels": 1,
              "n_train_frac": _n_train_frac,
              "n_val_frac": _n_val_frac},
    "protein": {"type": "classification",
                "num_classes": 2,
                "n_train_frac": _n_train_frac,
                "n_val_frac": _n_val_frac}
}

datasets_real = {"adult", "australian", "cifar10", "cod_rna", "covtype",
                 "emnist_balanced", "fashion_mnist", "iris",
                 "mnist", "protein"}


## Data generation procedures.

def get_data(dataset, rg=None):
    '''
    Takes a string, return a tuple of data and parameters.
    '''
    try:
        paras = dataset_paras[dataset]
    except KeyError:
        raise ValueError(
            "Did not recognize dataset {}.".format(dataset)
        )

    if dataset in datasets_real:
        return get_data_general(dataset=dataset, paras=paras, rg=rg)
    else:
        return get_data_simulated(paras=paras, rg=rg)


def get_data_simulated(paras, rg=None):
    '''
    Data generation function.
    This particular implementation is a simple noisy
    linear model.
    '''

    n_train = paras["n_train"]
    n_val = paras["n_val"]
    n_test = paras["n_test"]
    n_total = n_train+n_val+n_test
    d = paras["num_features"]
    
    ## Setup of random generator.
    if rg is None:
        ss = np.random.SeedSequence()
        rg = np.random.default_rng(seed=ss)
    
    ## Specifying the true underlying model.
    w_star = np.ones(d).reshape((d,1))
    true_model = LinearRegression(num_features=d,
                                  paras_init={"w":w_star})
    paras.update({"w_star": w_star})
    
    ## Noise generator and stats.
    noise_gen = get_generator(name=paras["noise_dist"],
                              rg=rg,
                              **paras["noise_paras"])
    noise_stats = get_stats(name=paras["noise_dist"],
                            rg=rg,
                            **paras["noise_paras"])
    paras.update({"noise_mean": noise_stats["mean"],
                  "noise_var": noise_stats["var"]})

    ## Data generation.
    X = rg.multivariate_normal(mean=np.zeros(d),
                               cov=paras["cov_X"],
                               size=n_total)
    noise = noise_gen(n=n_total).reshape((n_total,1))-noise_stats["mean"]
    y = true_model(X=X) + noise
    
    ## Split into appropriate sub-views and return.
    X_train = X[0:n_train,...]
    y_train = y[0:n_train,...]
    X_val = X[n_train:(n_train+n_val),...]
    y_val = y[n_train:(n_train+n_val),...]
    X_test = X[(n_train+n_val):,...]
    y_test = y[(n_train+n_val):,...]
    return (X_train, y_train, X_val, y_val, X_test, y_test, paras)


def get_data_general(dataset, paras, rg=None):
    '''
    General purpose data-getter.
    '''

    ## If not given a random generator, prepare one.
    if rg is None:
        ss = np.random.SeedSequence()
        rg = np.random.default_rng(seed=ss)

    ## Read the specified data.
    toread = os.path.join(dir_data_toread, dataset,
                          "{}.h5".format(dataset))
    with open_file(toread, mode="r") as f:
        print(f)
        X = f.get_node(where="/", name="X").read().astype(np.float32)
        y = f.get_node(where="/", name="y").read().astype(np.int64).ravel()
        print("Types: X ({}), y ({}).".format(type(X), type(y)))
    
    ## If sample sizes are correct, then get an index for shuffling.
    if len(X) != len(y):
        s_err = "len(X) != len(y) ({} != {})".format(len(X),len(y))
        raise ValueError("Dataset sizes wrong. "+s_err)
    else:
        idx_shuffled = rg.permutation(len(X))
    
    ## If specified, turn y into a one-hot format.
    try:
        if paras["type"] == "classification":
            y = onehot(y=y, num_classes=paras["num_classes"])
    except KeyError:
        None
    
    ## Do the actual shuffling.
    X = X[idx_shuffled,:]
    y = y[idx_shuffled]
    
    ## Normalize the inputs in a per-feature manner (as max/min are vecs).
    maxvec = np.max(X, axis=0)
    minvec = np.min(X, axis=0)
    X = X-minvec
    with np.errstate(divide="ignore", invalid="ignore"):
        X = X / (maxvec-minvec)
        X[X == np.inf] = 0
        X = np.nan_to_num(X)
    del maxvec, minvec

    ## Get split sizes (training, validation, testing).
    n_all, num_features = X.shape
    print("(n_all, num_features) = {}".format((n_all, num_features)))
    n_train = int(n_all*paras["n_train_frac"])
    n_val = int(n_all*paras["n_val_frac"])
    n_test = n_all-n_train-n_val
    print("n_train = {}".format(n_train))
    print("n_val = {}".format(n_val))
    print("n_test = {}".format(n_test))

    ## Learning task specific parameter additions.
    paras.update({"num_features": num_features})

    ## Do train/test split, with validation data if specified.
    X_train = X[0:n_train,...]
    y_train = y[0:n_train,...]
    if n_val > 0:
        X_val = X[n_train:(n_train+n_val),...]
        y_val = y[n_train:(n_train+n_val),...]
    else:
        X_val = None
        y_val = None
    X_test = X[(n_train+n_val):,...]
    y_test = y[(n_train+n_val):,...]
    
    print("Types and shapes:")
    print("X_train: {} and {}.".format(type(X_train), X_train.shape))
    print("y_train: {} and {}.".format(type(y_train), y_train.shape))
    if n_val > 0:
        print("X_val: {} and {}.".format(type(X_val), X_val.shape))
        print("y_val: {} and {}.".format(type(y_val), y_val.shape))
    print("X_test: {} and {}.".format(type(X_test), X_test.shape))
    print("y_test: {} and {}.".format(type(y_test), y_test.shape), "\n")

    return (X_train, y_train, X_val, y_val,
            X_test, y_test, paras)
    

###############################################################################
