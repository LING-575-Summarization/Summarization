#!/bin/sh

# testing for sentence length

#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_minus1 1 -1 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py sent_length_minus1 > rouge_score/sent_length_minus1

#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_5 1 5 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py sent_length_5 > rouge_score/sent_length_5
#
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_10 1 10 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py sent_length_10 > rouge_score/sent_length_10
#
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_15 1 15 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py sent_length_15 > rouge_score/sent_length_15
#
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_20 1 20 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py sent_length_20 > rouge_score/sent_length_20
#
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_25 1 25 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py sent_length_25 > rouge_score/sent_length_25
#
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/sent_length_30 1 30 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py sent_length_30 > rouge_score/sent_length_30


#############################################
# testing for gram
#
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/unigram 1 20 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py unigram > rouge_score/unigram
#
#echo "finished with unigram"
#
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/bigram 2 20 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py bigram > rouge_score/bigram
#
#echo "finished with bigram"
#
#
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/trigram 3 20 0.01 0.01 False True True
#python3 src/utils/eval_rouge.py trigram > rouge_score/trigram
#
#echo "finished with trigram"

############################################

# testing gram and sent_length

# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/bigram_sl_15 2 15 0.01 0.01 False True True
# python3 src/utils/eval_rouge.py bigram_sl_15 > rouge_score/bigram_sl_15

# echo "finished with bigram sent_length 15"

# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/bigram_sl_25 2 25 0.01 0.01 False True True
# python3 src/utils/eval_rouge.py bigram_sl_25 > rouge_score/bigram_sl_25

# echo "finished with bigram sent_length 25"


# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/trigram_sl_15 3 15 0.01 0.01 False True True
# python3 src/utils/eval_rouge.py trigram_sl_15 > rouge_score/trigram_sl_15

# echo "finished with trigram sent_length 15"


# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/trigram_sl_25 3 25 0.01 0.01 False True True
# python3 src/utils/eval_rouge.py trigram_sl_25 > rouge_score/trigram_sl_25

# echo "finished with trigram sent_length 25"

##########################################

# all with uppercase

# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/unigram_sl_15_lower_false 1 15 0.01 0.01 False False True
# python3 src/utils/eval_rouge.py unigram_sl_15_lower_false > rouge_score/unigram_sl_15_lower_false

# echo "finished with unigram sent_length 15 lowercasing false"


# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/unigram_sl_25_lower_false 1 25 0.01 0.01 False False True
# python3 src/utils/eval_rouge.py unigram_sl_25_lower_false > rouge_score/unigram_sl_25_lower_false

# echo "finished with unigram sent_length 25 lowercasing false"


# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/bigram_sl_15_lower_false 2 15 0.01 0.01 False False True
# python3 src/utils/eval_rouge.py bigram_sl_15_lower_false > rouge_score/bigram_sl_15_lower_false

# echo "finished with bigram sent_length 15 lowercasing false"

# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/bigram_sl_25_lower_false 2 25 0.01 0.01 False False True
# python3 src/utils/eval_rouge.py bigram_sl_25_lower_false > rouge_score/bigram_sl_25_lower_false

# echo "finished with bigram sent_length 25 lowercasing false"

##########################################

# all without logs

# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/unigram_sl_15_log_false 1 15 0.01 0.01 False True False
# python3 src/utils/eval_rouge.py unigram_sl_15_log_false > rouge_score/unigram_sl_15_log_false

# echo "finished with unigram sent_length 15 log false"


# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/unigram_sl_25_log_false 1 25 0.01 0.01 False True False
# python3 src/utils/eval_rouge.py unigram_sl_25_log_false > rouge_score/unigram_sl_25_log_false

# echo "finished with unigram sent_length 25 log false"


# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/bigram_sl_15_log_false 2 15 0.01 0.01 False True False
# python3 src/utils/eval_rouge.py bigram_sl_15_log_false > rouge_score/bigram_sl_15_log_false

# echo "finished with bigram sent_length 15 log false"

# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/D4/bigram_sl_25_log_false 2 25 0.01 0.01 False True False
# python3 src/utils/eval_rouge.py bigram_sl_25_log_false > rouge_score/bigram_sl_25_log_false

# echo "finished with bigram sent_length 25 log false"


##########################################
# testing clustering hyperparameters on experiment G2 - decided median tf-idf is the best hyperparameters
# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/clustering/ordered_G2/median_tfidf_G2 1 20 1 0.1 False True True



##########################################
# Running Ablation on best model: ILP

# repeat original
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J0 1 25 0.7 0.01 False True True
# python3 src/utils/eval_rouge.py J0 > rouge_score/J0

# echo "finished with J0"

# no sentence length
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J1 1 -1 0.7 0.01 False True True
# python3 src/utils/eval_rouge.py J1 > rouge_score/J1

# echo "finished with J1"

# bigram
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J2 2 25 0.7 0.01 False True True
# python3 src/utils/eval_rouge.py J2 > rouge_score/J2

# echo "finished with J2"

# trigram
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J3 3 25 0.7 0.01 False True True
# python3 src/utils/eval_rouge.py J3 > rouge_score/J3

# echo "finished with J3"

# low tf
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J4 1 25 0.7 0.001 False True True
# python3 src/utils/eval_rouge.py J4 > rouge_score/J4

# echo "finished with J4"


# low idf
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J6 1 25 0.001 0.01 False True True
# python3 src/utils/eval_rouge.py J6 > rouge_score/J6

# echo "finished with J6"


# eliminate punctuation
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J8 1 25 0.7 0.01 True True True
# python3 src/utils/eval_rouge.py J8 > rouge_score/J8

# echo "finished with J8"


# no lowercasing
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J9 1 25 0.7 0.01 False False True
# python3 src/utils/eval_rouge.py J9 > rouge_score/J9

# echo "finished with J9"


# no logging
#./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/ablation/J10 1 25 0.7 0.01 False True False
# python3 src/utils/eval_rouge.py J10 > rouge_score/J10

# echo "finished with J10"

./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/clustering/F8_before_content_realization 1 25 0.7 0.01 False True True

echo "finished with F8 clustering but no content realization"





