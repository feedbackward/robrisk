#!/bin/bash

ALGO="SGD"
ALPHA="0.05"
BATCH="1"
DATA="ds_normal"
LOSS="quadratic"
MODEL="linreg"
EPOCHS="30"
TASK="default"
TRIALS="100"
STEP="1.0"

python "learn_driver.py" --algo="$ALGO" --alpha="$ALPHA" --batch-size="$BATCH" --data="$DATA" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"

