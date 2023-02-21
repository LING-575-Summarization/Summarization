#!/bin/sh


python3 src/information_ordering/order_summaries.py $1 $2 $3


#1: summary directory
#2: output directory
#3: json_file : e.g. devtest.json

# ./src/scripts/order_summaries_clustering.sh outputs/test_lp_outputs/test_order outputs/test_lp_outputs/tested_order