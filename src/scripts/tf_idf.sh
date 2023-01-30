#!/bin/bash

/nopt/python-3.6/bin/python3 src/content_selection/tf_idf.py $1 $2 $3

#1 : path to json file "output/[training/dev/test].json"
#2: tf_delta, default is 1
#3: idf_delta, default is 1