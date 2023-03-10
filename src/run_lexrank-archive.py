from vectorizer import *
from tqdm import tqdm
import json, os
from utils import docset_loader
from extraction_methods.lexrank import LexRankFactory
from dataclasses import dataclass
from extraction_methods import SentenceIndex, create_clusters
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()
detokenize = lambda x: detokenizer.detokenize(x)

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
    ignore_punctuation: bool = True


@dataclass
class LSAExpt(Experiment):
    idf_level: str = "doc"
    vector: str = 'lsa'
    ignore_punctuation: bool = True
    lowercase: bool = True


ARGS=TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7) # BEST

EXPERIMENTS = [
    ("ablation", TFExpt(idf_level="documset", ngram=1, threshold=0.15, min_jaccard_dist=0.7)),
    # (2, TFExpt(idf_level="documset", ngram=1, threshold=0.15, min_jaccard_dist=0.7, min_length=15)),
    # (3, TFExpt(idf_level="documset", ngram=1, threshold=0.15, min_jaccard_dist=0.7, delta_idf=1.)),
    # (4, TFExpt(idf_level="documset", ngram=1, threshold=0.15, min_jaccard_dist=0.7, content_realization=True)),
    # (5, TFExpt(idf_level="documset", ngram=1, threshold=0.15, min_jaccard_dist=0.7,  min_length=15, content_realization=True)),
    # (6, TFExpt(idf_level="documset", ngram=1, threshold=0.15, min_jaccard_dist=0.7,  delta_idf=1, content_realization=True))
    # (1, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7)), #2
    # (2, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, content_realization=True)), #2
    # ("cr-test", TFExpt(idf_level="doc", ngram=1, threshold=0.15, min_jaccard_dist=0.7, content_realization=True)), #1
    # (3, BERTExpt(threshold=0.15, min_jaccard_dist=0.7)), 
    # (4, BERTExpt(threshold=0.15, min_jaccard_dist=0.8)), 
    # (6, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, content_realization=True)), 
    # (7, BERTExpt(threshold=0.15, min_jaccard_dist=0.7, content_realization=True)),
    # (8, BERTExpt(threshold=0.15, min_jaccard_dist=0.8, content_realization=True)),
    # (9, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=10)), 
    # (10, BERTExpt(threshold=0.15, min_jaccard_dist=0.7, min_length=10)),
    # (11, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=10, content_realization=True)), 
    # (12, BERTExpt(threshold=0.15, min_jaccard_dist=0.7, min_length=10, content_realization=True)), 
    # (13, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=15, content_realization=True)), 
    # (14, BERTExpt(threshold=0.15, min_jaccard_dist=0.7, min_length=15, content_realization=True)), 
    # (4, LSAExpt(idf_level="documset", threshold=0.15, min_jaccard_dist=0.7)),
    # (5, LSAExpt(idf_level="documset", threshold=0.15, min_jaccard_dist=0.7, min_length=10)),
    # (6, LSAExpt(idf_level="documset", threshold=0.15, min_jaccard_dist=0.7, min_length=10, content_realization=True)),
    # (7, LSAExpt(idf_level="documset", threshold=0.15, min_jaccard_dist=0.7, content_realization=True)),
    # (8, TFExpt(idf_level="documset", ngram=1, delta_idf=1., log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=10)),
    # (9, TFExpt(idf_level="documset", ngram=1, delta_idf=1., log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=20)),
    # (10, TFExpt(idf_level="documset", ngram=1, delta_idf=1., log_tf=True, threshold=0.15, min_jaccard_dist=0.7, min_length=10, content_realization=True)),
    # (11, TFExpt(idf_level="documset", ngram=1, delta_idf=0.7, log_tf=False, threshold=0.15, min_jaccard_dist=0.7, min_length=20, content_realization=True)),
]


def main():
    datafile = 'data/devtest.json'
    fractional_order = SentenceIndex(datafile)
    with open(datafile, 'r') as testdata:
        data = json.load(testdata).keys()
    for i, expt in EXPERIMENTS:
        vector_type = expt.pop('vector')
        realization = expt.pop('content_realization')
        LexRank = LexRankFactory(vector_type)
        print(f"Experiment ({vector_type}) {i}...")
        args = expt.as_dict()
        m = args.pop('idf_level', "doc")
        idf_docset = False if m == "doc" else True
        if idf_docset:
            lx = LexRank.from_data(datafile=[datafile, 'data/training.json'], sentences_are_documents=True,
                                    do_evaluate=False, **args)
            for docset_id in tqdm(data, desc="Evaluating documents"):
                docset, indices = docset_loader(
                    datafile, docset_id, sentences_are_documents=True)
                lx = lx.replace_evaldocs(docset, indices)
                if realization:
                    _docset, indices = docset_loader(datafile, docset_id)
                    result, order = lx.obtain_summary(_docset, coreference_resolution = True, detokenize=False, return_original=True)
                else:
                    result, order = lx.obtain_summary(coreference_resolution = False, detokenize=False, return_original=True)
                _order = [(i, s) for i, s in enumerate(order)]
                ordered = create_clusters(docset_id, order, fractional_order, datafile)
                # recover order
                s_order = []
                for s_o in ordered:
                    for _s in _order:
                        if _s[1] == s_o:
                            s_order.append(_s[0])
                            break
                result = [result[i] for i in s_order]
                id0 = docset_id[0:5]
                id1 = docset_id[-3]
                output_file = os.path.join('../bin', 'D6-lexrank', f'{id0}-A.M.100.{id1}.{i}')
                with open(output_file, 'w', encoding='utf8') as outfile:
                    for sentence in result:
                        outfile.write(detokenize(sentence))
                        outfile.write("\n")
        else:
            for docset_id in tqdm(data, desc="Evaluating documents"):
                lx = LexRank.from_data(datafile=datafile, documentset=docset_id,
                    sentences_are_documents=True, **args)
                if realization:
                    _docset, indices = docset_loader(datafile, docset_id)
                    result, order = lx.obtain_summary(_docset, coreference_resolution = True, detokenize=True, return_original=True)
                else:
                    result, order = lx.obtain_summary(coreference_resolution = False, detokenize=True, return_original=True)
                ordered = create_clusters(docset_id, order, fractional_order, datafile)
                _order = [(i, s) for i, s in enumerate(order)]
                ordered = create_clusters(docset_id, order, fractional_order, datafile)
                # recover order
                s_order = []
                for s_o in ordered:
                    for _s in _order:
                        if _s[1] == s_o:
                            s_order.append(_s[0])
                            break
                result = [result[i] for i in s_order]
                id0 = docset_id[0:5]
                id1 = docset_id[-3]
                output_file = os.path.join('../bin', 'D6-lexrank', f'{id0}-A.M.100.{id1}.{i}')
                with open(output_file, 'w', encoding='utf8') as outfile:
                    for sentence in result:
                        outfile.write(detokenize(sentence))
                        outfile.write("\n")


if __name__ == '__main__':
    main()