#!/bin/sh


#/nopt/python-3.6/bin/python3 src/content_selection/linear_programming.py $1 $2 > output/ILP_solver_output
python3 src/extraction_methods/linear_programming.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}> outputs/ILP_solver_output

# 1: json_file path for data: "output/[training/dev/test].json"
# 2: max summary length, default is 100
# 3: output_file directory
# 4: ngram
# 5: sent_length
# 6: delta_idf
# 7: delta_tf
# 8: eliminate_punctuation
# 9: lower_casing
# 10: log

# output/ILP_solver_output catches pulp's output for solving the ILP problem

# to run: ./src/scripts/linear_programming.sh output/[training/devtest/test].json 100

# example:

#
# ./src/scripts/linear_programming.sh outputs/devtest.json 100 outputs/test_lp_outputs 1 1000000 1 1 False True True
