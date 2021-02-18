#!/bin/bash

ALGO="SGD-RB"
ALPHA="0.05"
BATCH="1"
DATA="ds_lognormal"
LOSS="quadratic"
MODEL="linreg"
EPOCHS="40"
PROCS="10"
TASK="default"
TRIALS="100"
STEP="0.01"

python "learn_rb_driver.py" --algo="$ALGO" --alpha="$ALPHA" --batch-size="$BATCH" --data="$DATA" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-processes="$PROCS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"
