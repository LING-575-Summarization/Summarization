#!/bin/bash

dir="/dropbox/22-23/575x/Data/Documents"

src/scripts/./proc_docset.sh "$dir"/devtest/GuidedSumm10_test_topics.xml data/devtest --no_tokenize && \
    src/scripts/./proc_docset.sh "$dir"/training/2009/UpdateSumm09_test_topics.xml data/training --no_tokenize && \
    src/scripts/./proc_docset.sh "$dir"/evaltest/GuidedSumm11_test_topics.xml data/evaltest --no_tokenize
