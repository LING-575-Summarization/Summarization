#!/bin/bash

which conda

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/anaconda3/envs/575wi-env

which python

echo "Running LexRank..."
src/scripts/lexrank.sh --data_path "data" --data_set "devtest" --threshold 0.1 --error 1e-8
echo "Running Linear Programming..."
src/scripts/linear_programming.sh outputs/devtest.json 100
echo "Running LLM..."
src/scripts/./run_LLM.sh
