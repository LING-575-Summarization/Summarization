#!/bin/sh

echo "Running LexRank on devtest..."
./src/scripts/lexrank.sh --data_path data --data_set devtest --min_jaccard_dist 0.7 \
    --content_realization --idf_docset --ignore_punctuation --lowercase --ignore_stopwords
