#!/bin/bash

which conda

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/anaconda3/envs/575wi-env

which python

src/scripts/./run_LLM.sh