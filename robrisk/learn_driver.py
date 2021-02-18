'''Driver script for tests of CVaR-based learning algorithms.'''

## External modules.
import argparse
import json
import numpy as np
import os

## Internal modules.
from mml.utils import makedir_safe
from setup_algos import get_algo
from setup_data import get_data
from setup_eval import get_eval, eval_model, eval_write
from setup_losses import get_loss
from setup_models import get_model
from setup_results import results_dir
from setup_train import train_epoch


###############################################################################


## Basic setup.

parser = argparse.ArgumentParser(description="Arguments for driver script.")

parser.add_argument("--algo",
                    help="Algorithm class to test (default: SGD).",
                    type=str, default="SGD", metavar="S")
parser.add_argument("--alpha",
                    help="Set CVaR level to 1-alpha (default: 0.05).",
                    type=float, default=0.05, metavar="F")
parser.add_argument("--batch-size",
                    help="Mini-batch size for algorithms (default: 1).",
                    type=int, default=1, metavar="N")
parser.add_argument("--data",
                    help="Specify data set to be used (default: None).",
                    type=str, default=None, metavar="S")
parser.add_argument("--loss",
                    help="Loss name. (default: quadratic)",
                    type=str, default="quadratic", metavar="S")
parser.add_argument("--model",
                    help="Model class. (default: linreg)",
                    type=str, default="linreg", metavar="S")
parser.add_argument("--no-cvar",
                    help="Turn off use of CVaR loss (default: False).",
                    action="store_true", default=False)
parser.add_argument("--num-epochs",
                    help="Number of epochs to run (default: 3)",
                    type=int, default=3, metavar="N")
parser.add_argument("--num-trials",
                    help="Number of independent random trials (default: 1)",
                    type=int, default=1, metavar="N")
parser.add_argument("--step-size",
                    help="Step size parameter (default: 0.01)",
                    type=float, default=0.01, metavar="F")
parser.add_argument("--task-name",
                    help="A task name. Default is the word default.",
                    type=str, default="default", metavar="S")

## Setup of random generator.
ss = np.random.SeedSequence()
rg = np.random.default_rng(seed=ss)

## Parse the arguments passed via command line.
args = parser.parse_args()
if args.data is None:
    raise TypeError("Given --data=None, should be a string.")

## Name to be used identifying the results etc. of this experiment.
towrite_name = args.task_name+"-"+"_".join([args.model, args.algo])

## Prepare a directory to save results.
towrite_dir = os.path.join(results_dir, args.data)
makedir_safe(towrite_dir)


## Main process.
if __name__ == "__main__":

    ## Arguments for losses.
    loss_kwargs = {"alpha": args.alpha,
                   "use_cvar": args.no_cvar == False}
    
    ## Arguments for algorithms.
    algo_kwargs = {}
    
    ## Arguments for models.
    model_kwargs = {}
    
    ## Prepare the loss for training.
    loss = get_loss(name=args.loss, **loss_kwargs)
    
    ## Start the loop over independent trials.
    for trial in range(args.num_trials):
        
        ## Load in data.
        print("Doing data prep.")
        (X_train, y_train, X_val, y_val,
         X_test, y_test, ds_paras) = get_data(dataset=args.data, rg=rg)
        
        ## Data index.
        data_idx = np.arange(len(X_train))
        
        ## Prepare evaluation metric(s).
        eval_dict = get_eval(loss_name=args.loss,
                             model_name=args.model,
                             **loss_kwargs, **ds_paras)
        
        ## Model setup.
        model = get_model(name=args.model,
                          paras_init=None,
                          rg=rg,
                          **loss_kwargs, **model_kwargs, **ds_paras)
        
        ## Prepare algorithms.
        model_dim = model.paras["w"].size
        algo_kwargs.update(
            {"num_data": len(X_train),
             "step_size": args.step_size/np.sqrt(model_dim)}
        )
        algo = get_algo(
            name=args.algo,
            model=model,
            loss=loss,
            **ds_paras, **algo_kwargs
        )
        
        ## Prepare storage for performance evaluation this trial.
        store_train = {
            key: np.zeros(shape=(args.num_epochs,1),
                          dtype=np.float32) for key in eval_dict.keys()
        }
        if X_test is not None:
            store_test = {
                key: np.zeros_like(
                    store_train[key]
                ) for key in eval_dict.keys()
            }
        else:
            store_test = {}
        storage = (store_train, store_test)
        
        ## Loop over epochs, done in the parent process.
        for epoch in range(args.num_epochs):
            
            print("(Tr {}) Ep {} starting.".format(trial, epoch))
            
            ## Shuffle data.
            rg.shuffle(data_idx)
            X_train = X_train[data_idx,...]
            y_train = y_train[data_idx,...]

            ## Carry out one epoch's worth of training.
            train_epoch(algo=algo,
                        loss=loss,
                        X=X_train,
                        y=y_train,
                        batch_size=args.batch_size)

            ## Evaluate performance of the sub-process candidates.
            eval_model(epoch=epoch,
                       model=model,
                       storage=storage,
                       data=(X_train,y_train,X_test,y_test),
                       eval_dict=eval_dict)
            
            print("(Tr {}) Ep {} finished.".format(trial, epoch), "\n")


        ## Write performance for this trial to disk.
        perf_fname = os.path.join(towrite_dir,
                                  towrite_name+"-"+str(trial))
        eval_write(fname=perf_fname,
                   storage=storage)

    ## Write a JSON file to disk that summarizes key experiment parameters.
    dict_to_json = vars(args)
    dict_to_json.update({
        "entropy": ss.entropy # for reproducability.
    })
    towrite_json = os.path.join(towrite_dir, towrite_name+".json")
    with open(towrite_json, "w", encoding="utf-8") as f:
        json.dump(obj=dict_to_json, fp=f,
                  ensure_ascii=False,
                  sort_keys=True, indent=4)


###############################################################################
