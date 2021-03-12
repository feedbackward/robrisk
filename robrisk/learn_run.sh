#!/bin/bash

ALGO_ANCILLARY="SGD"
ALGO_MAIN=""
ALPHA="0.05"
BATCH="0"
DATA="adult"
ENTROPY="256117190779556056928268872043329970341"
LOSS="logistic"
MODEL="linreg_multi"
EPOCHS="30"
TASK="default"
TRIALS="10"
STEP="0.1"

python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --alpha="$ALPHA" --batch-size="$BATCH" --data="$DATA" --entropy="$ENTROPY" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"
