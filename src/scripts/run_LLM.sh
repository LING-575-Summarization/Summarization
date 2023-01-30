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
python src/LLM/preprocess.py --raw_json_dir "data/"