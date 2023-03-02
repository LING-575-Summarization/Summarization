from vectorizer import *
from tqdm import tqdm
import json, os
import sys
from content_selection.lexrank import LexRankFactory
from dataclasses import dataclass

@dataclass
class Experiment:
    idf_level: str
    ngram: int
    delta_idf: float
    log_tf: bool

    def as_dict(self):
        x = self.__dict__
        m = x.pop('idf_level')
        l = False if m == "sentence" else True
        return x, l


EXPT=Experiment("sentence", 1, 0.7, False)

# EXPERIMENTS = [
#     # Experiment("sentence", 1, 0., False),
#     # Experiment("documset", 1, 0., False),
#     # Experiment("sentence", 2, 0., False),
#     # Experiment("documset", 2, 0., False),
#     #  Experiment("sentence", 1, 0.7, False),
#     # Experiment("documset", 1, 0.7, False),
#     Experiment("sentence", 1, 0., True),
#     Experiment("documset", 1, 0., True),
# ]

def main():
    with open(DATASET, 'r') as datafile:
        data = json.load(datafile).keys()
    LexRank = LexRankFactory('tfidf')
    args, _ = EXPT.as_dict()  
    for docset_id in tqdm(data, desc="Evaluating documents"):
        lx = LexRank.from_data(datafile=DATASET, documentset=docset_id,
            sentences_are_documents=True, min_length=5, min_jaccard_dist=0.6, **args)
        result = lx.obtain_summary(detokenize=True)
        id0 = docset_id[0:5]
        id1 = docset_id[-3]
        output_file = os.path.join('outputs', 'D4', f'{id0}-A.M.100.{id1}.2')
        with open(output_file, 'w') as outfile:
            outfile.write(result)


if __name__ == '__main__':
    DATASET = sys.argv[1]
    main()