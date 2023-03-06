from vectorizer import *
from tqdm import tqdm
import json, os
from utils import docset_loader
from extraction_methods.lexrank import LexRankFactory
from content_realization import replace_referents
from dataclasses import dataclass

@dataclass
class Experiment:
    threshold: float = 0.
    error: float = 1e-16
    min_length: int = 7
    min_jaccard_dist: float = 0.6
    content_realization: bool = False

    def as_dict(self):
        x = self.__dict__
        return x

    def pop(self, key: str):
        value = self.__dict__[key]
        delattr(self, key)
        return value


@dataclass
class W2VExpt(Experiment):
    vector: str = 'word2vec'
    reduction: str = 'centroid'


@dataclass
class BERTExpt(Experiment):
    vector: str = 'bert'


@dataclass
class TFExpt(Experiment):
    idf_level: str = "doc"
    ngram: int = 1
    delta_idf: float = 0.
    log_tf: bool = False
    vector: str = 'tfidf'
    ignore_punctuation: bool = False


@dataclass
class LSAExpt(Experiment):
    idf_level: str = "doc"
    vector: str = 'lsa'
    ignore_punctuation: bool = False
    lowercase: bool = True


ARGS=TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7) # BEST

EXPERIMENTS = [
    # (1, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7)), #2
    # (2, TFExpt(idf_level="documset", ngram=1, delta_idf=1., log_tf=True, threshold=0.15, min_jaccard_dist=0.7)), #1
    # (3, BERTExpt(threshold=0.15, min_jaccard_dist=0.7)), 
    # (4, BERTExpt(threshold=0.15, min_jaccard_dist=0.8)), 
    # (5, TFExpt(idf_level="documset", ngram=1, delta_idf=1., log_tf=True, threshold=0.15, min_jaccard_dist=0.7, content_realization=True)), 
    # (6, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, content_realization=True)), 
    # (7, BERTExpt(threshold=0.15, min_jaccard_dist=0.7, content_realization=True)),
    # (8, BERTExpt(threshold=0.15, min_jaccard_dist=0.8, content_realization=True)),
    # (9, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=10)), 
    # (10, BERTExpt(threshold=0.15, min_jaccard_dist=0.7, min_length=10))
    # (11, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=10, content_realization=True)), 
    # (12, BERTExpt(threshold=0.15, min_jaccard_dist=0.7, min_length=10, content_realization=True)), 
    # (13, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=15, content_realization=True)), 
    # (14, BERTExpt(threshold=0.15, min_jaccard_dist=0.7, min_length=15, content_realization=True)), 
    (15, LSAExpt(threshold=0.15, min_jaccard_dist=0.7, min_length=15)),
]


def main():
    with open('data/devtest.json', 'r') as datafile:
        data = json.load(datafile).keys()
    for i, expt in EXPERIMENTS:
        vector_type = expt.pop('vector')
        realization = expt.pop('content_realization')
        LexRank = LexRankFactory(vector_type)
        print(f"Experiment ({vector_type}) {i}...")
        args = expt.as_dict()
        m = args.pop('idf_level', "doc")
        idf_docset = False if m == "doc" else True
        if idf_docset:
            lx = LexRank.from_data(datafile='data/devtest.json', sentences_are_documents=True,
                                    do_evaluate=False, **args)
            for docset_id in tqdm(data, desc="Evaluating documents"):
                docset, indices = docset_loader(
                    'data/devtest.json', docset_id, sentences_are_documents=True)
                lx = lx.replace_evaldocs(docset, indices)
                if realization:
                    _docset, indices = docset_loader('data/devtest.json', docset_id)
                    result = lx.obtain_summary(_docset, coreference_resolution = True, detokenize=True)
                else:
                    result = lx.obtain_summary(coreference_resolution = False, detokenize=True)
                id0 = docset_id[0:5]
                id1 = docset_id[-3]
                output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i}')
                with open(output_file, 'w', encoding='utf8') as outfile:
                    outfile.write(result)
        else:
            for docset_id in tqdm(data, desc="Evaluating documents"):
                lx = LexRank.from_data(datafile='data/devtest.json', documentset=docset_id,
                    sentences_are_documents=True, **args)
                if realization:
                    _docset, indices = docset_loader('data/devtest.json', docset_id)
                    result = lx.obtain_summary(_docset, coreference_resolution = True, detokenize=True)
                else:
                    result = lx.obtain_summary(coreference_resolution = False, detokenize=True)
                id0 = docset_id[0:5]
                id1 = docset_id[-3]
                output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i}')
                with open(output_file, 'w', encoding='utf8') as outfile:
                    outfile.write(result)


if __name__ == '__main__':
    main()