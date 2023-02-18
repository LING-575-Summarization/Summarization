#!/bin/sh

# testing for sentence length

#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_minus1 1 -1 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py sent_length_minus1 > rouge_score/sent_length_minus1

./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_5 1 5 0.01 0.01 False True True
python3 src/utils/eval_rouge.py sent_length_5 > rouge_score/sent_length_5

./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_10 1 10 0.01 0.01 False True True
python3 src/utils/eval_rouge.py sent_length_10 > rouge_score/sent_length_10

./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_15 1 15 0.01 0.01 False True True
python3 src/utils/eval_rouge.py sent_length_15 > rouge_score/sent_length_15

./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_20 1 20 0.01 0.01 False True True
python3 src/utils/eval_rouge.py sent_length_20 > rouge_score/sent_length_20

./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_25 1 25 0.01 0.01 False True True
python3 src/utils/eval_rouge.py sent_length_25 > rouge_score/sent_length_25

./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_30 1 30 0.01 0.01 False True True
python3 src/utils/eval_rouge.py sent_length_30 > rouge_score/sent_length_30





