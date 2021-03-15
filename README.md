# robrisk: learning with diverse risk functions under potentially heavy-tailed feedback

This repository houses software for recreating the numerical experiments done in the following paper:

- Learning with risk-averse feedback under potentially heavy tails. Matthew J. Holland and El Mehdi Haress. *AISTATS 2021*.

The software here can be used to faithfully reproduce all the experimental results given in the above paper, and can also be easily applied to more general machine learning tasks, going well beyond the examples considered here.

A table of contents for this README file:

- <a href="#setup_init">Setup: initial software preparation</a>
- <a href="#start">Getting started</a>
- <a href="#demos">List of demos</a>
- <a href="#safehash">Safe hash values</a>


<a id="setup_init"></a>
## Setup: initial software preparation

To begin, please ensure you have the <a href="https://github.com/feedbackward/mml#prereq">prerequisite software</a> used in the setup of our `mml` repository.

Next, make a local copy of the repository and create a virtual environment for working in as follows:

```
$ git clone https://github.com/feedbackward/mml.git
$ git clone https://github.com/feedbackward/robrisk.git
$ git clone https://github.com/feedbackward/sgd-roboost.git
$ conda create -n robrisk python=3.8 jupyter matplotlib pip pytables scipy
$ conda activate robrisk
```

Having made (and activated) this new environment, we would like to use `pip` to install the supporting libraries for convenient access. This is done easily, by simply running

```
(robrisk) $ cd [mml path]/mml
(robrisk) $ pip install -e ./
(robrisk) $ cd [sgd-roboost path]/sgd-roboost
(robrisk) $ pip install -e ./
```

with the `[* path]` placeholders replaced with the path to wherever you used `clone` to copy the repositories to. If you desire a safe, tested version of `mml` and `sgd-roboost`, just run

```
(robrisk) $ git checkout [safe hash mml]
(robrisk) $ git checkout [safe hash sgd-roboost]
```

before the `pip install -e ./` commands above. The `[safe hash *]` placeholders are to be replaced using the <a href="#safehash">safe hash values</a> given at the end of this document.


<a id="start"></a>
## Getting started

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


<a id="safehash"></a>
## Safe hash values

- Replace `[safe hash mml]` with `1f6fa730e86ad7da88ba5b33400b0ec476d2cd1d`.
- Replace `[safe hash sgd-roboost]` with `4fc4fb5d129d0fbc93813d101ee141ad01fce1c4`

__Date of safe hash test:__ 2021/03/12.
