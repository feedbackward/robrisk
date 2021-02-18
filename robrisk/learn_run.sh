#!/bin/bash

ALGO="RGD-M"
ALPHA="0.05"
BATCH="0"
DATA="ds_lognormal"
LOSS="quadratic"
MODEL="linreg"
EPOCHS="40"
TASK="default"
TRIALS="100"
STEP="0.1"

python "learn_driver.py" --algo="$ALGO" --alpha="$ALPHA" --batch-size="$BATCH" --data="$DATA" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"
