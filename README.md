# robrisk: learning with diverse risk functions under potentially heavy-tailed feedback

This repository will houses code for recreating the numerical experiments done in the following paper:

- Learning with risk-averse feedback under potentially heavy tails. Matthew J. Holland and El Mehdi Haress. *AISTATS 2021*.

The software here can be used to faithfully reproduce all the experimental results given in the above paper, and can also be easily applied to more general machine learning tasks, going well beyond the examples considered here.

A table of contents for this README file:

- <a href="#code">Setup: software for numerical experiments</a>
- <a href="#demos">List of demos</a>

Before starting on the setup, we assume the following about the user's environment:

- has access to a `bash` shell
- can use `wget` to download datasets
- has `unzip`, `git`, and `conda` installed

and finally that the user has run

```
$ conda update -n base conda
```

This repository contains code which is "local" to the experiments done in the above papers, but it makes use of many functions which are of a much more general-purpose nature. Such functions are implemented seperately in our <a href="https://github.com/feedbackward/mml">mml</a> and <a href="https://github.com/feedbackward/sgd-roboost">sgd-roboost</a> repositories (details given below).

Let us now proceed to the software setup.


<a id="code"></a>
## Setup: software for numerical experiments

We begin with software that is directly related to the demos of interest. We proceed assuming that the user is working in a directory that does *not* contain directories with the names of those we are about to `clone` (i.e., does __not__ contain `robrisk`, `mml`, or `sgd-roboost`).

With these prerequisites understood and in place, the clerical setup is quite easy. The main initial steps are as follows.

```
$ git clone https://github.com/feedbackward/robrisk.git
$ git clone https://github.com/feedbackward/mml.git
$ git clone https://github.com/feedbackward/sgd-roboost.git
$ conda create -n robrisk python=3.8 jupyter matplotlib pip pytables scipy
$ conda activate robrisk
(robrisk) $ cd mml
(robrisk) $ git checkout [SHA mml]
(robrisk) $ pip install -e ./
(robrisk) $ cd ../sgd-roboost
(robrisk) $ git checkout [SHA sgd-roboost]
(robrisk) $ pip install -e ./
(robrisk) $ cd ../robrisk
```

For the `[SHA *]` placeholders, the following are safe, tested values (tested 2021/02/22):

- For `[SHA mml]`, replace with `6ee6a8f924b610ccfb9c5239e852ac7df72cf14c`.
- For `[SHA sgd-roboost]`, replace with `47cc2e9e4b240590bc3ed369086c8e78bba194a7`.

One __important__ clerical step is to modify the variable `todo_roboost` in `sgd-roboost/roboost/setup_roboost.py`; the default is a list of many different methods, but for our purposes here, we will only use (and indeed only allow) one, although any one is fine. The default used in all our relevant experiments is as follows:

```
todo_roboost = ["valid-robust"]
```

With this preparation in place, we can get to running the experiments. At a high level, we have basically three types of files:

- __Setup files:__ these take the form `setup_*.py`.
  - Configuration for all elements of the learning process, with one setup file for each of the following major categories: learning algorithms, data preparation, learned model evaluation, loss functions, models, result processing, and general-purpose training functions.

- __Driver scripts:__ these take the form `learn*_driver.py`.
  - These scripts control the flow of the learning procedure and handle all the clerical tasks such as organizing, naming, and writing numerical results to disk. No direct modification to these scripts is needed to run the experiments in the above paper.

- __Execution scripts:__ these take the form `learn*_run.sh`.
  - The choice of algorithm, model, data generation protocol, among other key parameters is made within these simple shell scripts. See the demo notebook for more details.

Additional details will be provided in the demo notebook provided. The overall flow is quite straightforward. Running a script `learn*_run.sh` in `bash`, the corresponding Python driver script `learn*_driver.py` is passed all the key experimental parameters as arguments. Results are written to disk with the following nomenclature:

```
[results_dir]/[data]/[task]-[model]_[algo]-[trial].[descriptor]
```

The `descriptor` depends on the evaluation metrics used, all specified in `setup_eval.py`. As for the rest of the elements in the same, these are determined completely by the experimental parameters passed via the execution script run. Please see the demo notebook linked below for more detailed information.

Finally, we note that there are two minor differences between the tests implemented here and the tests done in the original version of the paper cited above. First is that here we only record performance once every epoch, instead of multiple times per epoch. Second, in the original tests, we randomly generated one test set *before* running the trials in which fresh training data was randomly generated. In contrast, the tests here generate both the training and testing data fresh for each trial.


<a id="demos"></a>
## List of demos

This repository includes detailed demonstrations to walk the user through re-creating the results in the paper cited at the top of this document. Below is a list of demo links which give our demos (constructed in Jupyter notebook form) rendered using the useful <a href="https://github.com/jupyter/nbviewer">nbviewer</a> service.

- <a href="https://nbviewer.jupyter.org/github/feedbackward/robrisk/blob/main/robrisk/demo_static.ipynb">Demo: competing methods for CVaR estimation</a>
- <a href="https://nbviewer.jupyter.org/github/feedbackward/robrisk/blob/main/robrisk/demo_dynamic.ipynb">Demo: empirical analysis of CVaR-based learning</a>
