#!/bin/bash

ALGO_ANCILLARY="SGD"
ALGO_MAIN="Ave"
ALPHA="0.05"
BATCH="1"
DATA="ds_lognormal"
ENTROPY="256117190779556056928268872043329970341"
LOSS="quadratic"
MODEL="linreg"
EPOCHS="40"
PROCS="10"
TASK="default"
TRIALS="100"
STEP="0.01"

python "learn_rb_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --alpha="$ALPHA" --batch-size="$BATCH" --data="$DATA" --entropy="$ENTROPY" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-processes="$PROCS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"
