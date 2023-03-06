#!/bin/bash

conda env create -f env/hilly-env.yml

conda activate summarization

echo "pre-processing data"
src/scripts/open_all_files.sh

echo "Running Linear Programming on devtest..."
./src/scripts/linear_programming.sh data/devtest.json 100 outputs/D5_devtest 1 25 0.7 0.01 False True True
echo "Running Linear Programming on evaltest..."
./src/scripts/linear_programming.sh data/evaltest.json 100 outputs/evaltest 1 25 0.7 0.01 False True True


# echo "Running LexRank..."
# src/scripts/lexrank.sh --data_path "data" --data_set "devtest" --threshold 0.1 --error 1e-8

# echo "Running LLM..."
# src/scripts/./run_LLM.sh

# run ROUGE on devtest and evaltest files
