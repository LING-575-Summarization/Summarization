#!/bin/sh


#/nopt/python-3.6/bin/python3 src/content_selection/linear_programming.py $1 $2 > output/ILP_solver_output
python3 src/content_selection/linear_programming.py $1 $2 > output/ILP_solver_output

# 1: json_file path for data: "output/[training/dev/test].json"
# 2: max summary length, default is 100
# output/ILP_solver_output catches pulp's output for solving the ILP problem

# to run: ./src/scripts/linear_programming.sh output/[training/devtest/test].json 100
