#!/bin/bash

which conda

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/anaconda3/envs/575wi-env

which python

echo "Running LLM..."
src/scripts/./run_LLM.sh
echo "Running LexRank..."
/nopt/python-3.6/bin/python3 src/content_selection/lexrank.py --data_path "data" --data_set "devtest" --threshold 0.1 --error 1e-8
echo "Running Linear Programming..."
src/scripts/linear_programming.sh outputs/devtest.json 100
