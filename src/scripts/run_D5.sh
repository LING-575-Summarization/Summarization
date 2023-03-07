#!/bin/bash

conda env create -f src/environment.yml

conda activate summarization

python src/tests/checkinstall.py

echo "pre-processing data"
src/scripts/open_all_files.sh

echo "Running Baseline Top K on devtest..."
# ./src/scripts/baseline.sh data/devtest.json 100 outputs/D5_devtest

echo "Running Baseline Top K on evaltest..."
# ./src/scripts/baseline.sh data/evaltest.json 100 outputs/D5_evaltest

echo "Running Linear Programming on devtest..."
./src/scripts/linear_programming.sh data/devtest.json 100 outputs/D5_devtest 1 25 0.7 0.01 False True True
echo "Running Linear Programming on evaltest..."
./src/scripts/linear_programming.sh data/evaltest.json 100 outputs/D5_evaltest 1 25 0.7 0.01 False True True


# echo "Running LexRank..."
# src/scripts/lexrank.sh --data_path "data" --data_set "devtest" --threshold 0.1 --error 1e-8

echo "Running LLM..."
src/scripts/./run_LLM.sh

# run ROUGE on devtest and evaltest files
