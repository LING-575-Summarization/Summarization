from vectorizer import *
from tqdm import tqdm
import json, os
from utils import docset_loader, flatten_list, detokenize_list_of_tokens
from content_selection.lexrank import LexRankFactory
from content_realization import replace_referents
from dataclasses import dataclass

@dataclass
class Experiment:
    threshold: float = 0.
    error: float = 1e-16
    min_length: int = 7
    min_jaccard_dist: float = 0.6

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
    idf_level: str = "sentence"
    ngram: int = 1
    delta_idf: float = 0.
    log_tf: bool = False
    vector: str = 'tfidf'
    ignore_punctuation: bool = False


ARGS=TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7) # BEST

EXPERIMENTS = [
    (1, TFExpt(idf_level="sentence", ngram=1, delta_idf=0., log_tf=False)),
    (2, TFExpt(idf_level="sentence", ngram=1, delta_idf=0.7, log_tf=False)),
    # (3, TFExpt(idf_level="sentence", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7)),
    # (4, TFExpt(idf_level="sentence", ngram=1, delta_idf=0., log_tf=False, threshold=0.15, min_jaccard_dist=0.7)),
    # (5, TFExpt(idf_level="sentence", ngram=1, delta_idf=0., log_tf=True)),
    # (6, TFExpt(idf_level="sentence", ngram=1, delta_idf=1., log_tf=True)),
    # (7, TFExpt(idf_level="sentence", ngram=1, delta_idf=1., log_tf=True, threshold=0.15, min_jaccard_dist=0.7)),
    # (8, TFExpt(idf_level="sentence", ngram=1, delta_idf=0., log_tf=True, threshold=0.15, min_jaccard_dist=0.7)),
    # (9, TFExpt(idf_level="sentence", ngram=2, delta_idf=1., log_tf=True)),
    # (10, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False)),
    # (11, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7)), #4
    # (12, TFExpt(idf_level="documset", ngram=1, delta_idf=1., log_tf=True)),
    # (13, TFExpt(idf_level="documset", ngram=1, delta_idf=1., log_tf=True, threshold=0.15, min_jaccard_dist=0.7)), #3
    # (14, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=True, threshold=0.15, min_jaccard_dist=0.7)), #1
    # (15, W2VExpt(threshold=0.15, min_jaccard_dist=0.7)), #5
    # (16, BERTExpt(threshold=0.15, min_jaccard_dist=0.7)), #2
    # (17, W2VExpt(threshold=0.3, min_jaccard_dist=0.7)),
    # (18, BERTExpt(threshold=0.3, min_jaccard_dist=0.7)),
    # (19, W2VExpt(reduction='normalized_mean'))
]

EXPERIMENTS2 = [
    (20, W2VExpt(reduction='normalized_mean', threshold=0.15, min_jaccard_dist=0.7)),
    (21, BERTExpt(threshold=0.15, min_jaccard_dist=0.8)),
    (22, W2VExpt(threshold=0.15, min_jaccard_dist=0.7)),
    (23, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15)),
    (24, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.8)),
    (25, TFExpt(idf_level="documset", ngram=2, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7))
]

EXPERIMENTS3 = [
    # (11, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7)),
    # (14, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=True, threshold=0.15, min_jaccard_dist=0.7)),
    # (26, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, ignore_punctuation=True, threshold=0.15, min_jaccard_dist=0.7)),
    # (27, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=True, ignore_punctuation=True, threshold=0.15, min_jaccard_dist=0.7)),
    (28, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.8)),
]

def main():
    with open('data/devtest.json', 'r') as datafile:
        data = json.load(datafile).keys()
    for i, expt in EXPERIMENTS:
        vector_type = expt.pop('vector')
        LexRank = LexRankFactory(vector_type)
        print(f"Experiment ({vector_type}) {i}...")
        args = expt.as_dict()
        m = args.pop('idf_level', "sentence")
        idf_docset = False if m == "sentence" else True
        if idf_docset:
            lx = LexRank.from_data(datafile='data/devtest.json', sentences_are_documents=True,
                                    do_evaluate=False, **args)
            for docset_id in tqdm(data, desc="Evaluating documents"):
                docset, indices = docset_loader(
                    'data/devtest.json', docset_id)
                as_sentences = flatten_list([flatten_list(doc) for doc in docset])
                for j in range(len(as_sentences)):
                    index_key = docset_id + "." + str(j)
                    indices[index_key] = as_sentences[j]
                lx = lx.replace_evaldocs(as_sentences, indices)
                result = lx.obtain_summary(detokenize=False)
                print("before nounreplace")
                print(detokenize_list_of_tokens(result))
                result = replace_referents(result, docset)
                print("after nounreplace")
                print(result)
                id0 = docset_id[0:5]
                id1 = docset_id[-3]
                output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i}')
                with open(output_file, 'w') as outfile:
                    outfile.write(result)
        else:
            for docset_id in tqdm(data, desc="Evaluating documents"):
                docset, _ = docset_loader(
                    'data/devtest.json', docset_id)
                as_sentences = flatten_list([flatten_list(doc) for doc in docset])
                indices = []
                for j in range(len(as_sentences)):
                    index_key = docset_id + "." + str(j)
                    indices[index_key] = as_sentences[j]
                lx = LexRank(as_sentences, indices, **args)
                print("before nounreplace")
                result = lx.obtain_summary(detokenize=False)
                print(detokenize_list_of_tokens(result))
                print("after nounreplace")
                result = replace_referents(result, docset)
                print(result)
                id0 = docset_id[0:5]
                id1 = docset_id[-3]
                output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i}')
                with open(output_file, 'w') as outfile:
                    outfile.write(result)


if __name__ == '__main__':
    main()