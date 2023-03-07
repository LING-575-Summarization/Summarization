#!/bin/bash

which conda

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/anaconda3/envs/575wi-env

which python

# Get First JSON via proc_docset
python src/preprocess/proc_docset.py "/dropbox/22-23/575x/Data/Documents/devtest/GuidedSumm10_test_topics.xml" data/devtest --gold_directory "/home2/junyinc/dropbox/22-23/575x/Data/models/devtest/" --no_tokenize
python src/preprocess/proc_docset.py "/dropbox/22-23/575x/Data/Documents/training/2009/UpdateSumm09_test_topics.xml" data/training --gold_directory "/home2/junyinc/dropbox/22-23/575x/Data/models/training/2009/" --no_tokenize
python src/preprocess/proc_docset.py "/dropbox/22-23/575x/Data/Documents/evaltest/GuidedSumm11_test_topics.xml" data/evaltest --gold_directory "/home2/junyinc/dropbox/22-23/575x/Data/models/evaltest/" --no_tokenize


# Get Second JSON via LLM preprocess
python src/LLM/Summarize/preprocess.py --raw_json_dir "data/" --do_mask --do_previous_mask --do_throw --rouge "rouge1"
python src/LLM/Summarize/pipeline.py --batch_size 1 --dataset_type "train" --output_dir "outputs/D5_devtest"
python src/LLM/Summarize/pipeline.py --batch_size 1 --dataset_type "validation" --output_dir "outputs/D5_evaltest"
