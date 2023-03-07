from vectorizer import *
from tqdm import tqdm
import json, os
from utils import docset_loader
from extraction_methods.lexrank import LexRankFactory
from content_realization import replace_referents
from dataclasses import dataclass
from extraction_methods import SentenceIndex, create_clusters
from argparse import ArgumentParser

def get_args():
    argparser = ArgumentParser()
    argparser.add_argument(
        '--data_path', '-dpath', type=str, default='data',
        help='Path to document set directory'
    )
    argparser.add_argument(
        '--data_set', '-ds', type=str, required=True,
        help='Whether to use the test, dev, or train data split'
    )
    argparser.add_argument(
        '--threshold', '-t', type=float, required=True,
        help='The threshold to use when creating weighted graph'
    )
    argparser.add_argument(
        '--error', '-e', type=float, required=True,
        help='The error to use when solving for the eigenvalue'
    )
    argparser.add_argument(
        "--min_length", type=int, default=7,
    )
    argparser.add_argument(
        "--min_jaccard_dist", type=float, default=0.6,
    )
    argparser.add_argument(
        "--content_realization",  action='store_true'
    )
    argparser.add_argument(
        "--docset_idf",  action='store_true'
    )
    argparser.add_argument(
        "--ngrams", type=int, default=1,
    )
    argparser.add_argument(
        "--log_tf", action='store_true'
    )
    argparser.add_argument(
        "--ignore_punctuation", action='store_true'
    )
    argparser.add_argument(
        "--lowercase", action='store_true'
    )
    argparser.add_argument(
        "--remove_stopwords", action='store_true'
    )
    args, _ = argparser.parse_known_args()
    return args


@dataclass
class LexRankArgs:
    threshold: float = 0.
    error: float = 1e-16
    min_length: int = 7
    min_jaccard_dist: float = 0.6
    content_realization: bool = True
    idf_level: str = "documset"
    ngram: int = 1
    delta_idf: float = 0.
    log_tf: bool = False
    vector: str = 'tfidf'
    ignore_punctuation: bool = True
    ignore_stopwords: bool = True
    lowercase: bool = True
    

def main():
    args = get_args()
    dataset = args.data_set
    datafile = f'data/{dataset}.json'
    fractional_order = SentenceIndex(datafile)
    with open(datafile, 'r') as datafile:
        data = json.load(datafile).keys()

    l_args = vars(args)
    l_args.pop('data_path')
    l_args.pop('data_set')

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
                result = lx.obtain_summary(_docset, coreference_resolution = True, detokenize=False)
            else:
                result = lx.obtain_summary(coreference_resolution = False, detokenize=False)
            result = create_clusters(docset_id, result, fractional_order, datafile)
            id0 = docset_id[0:5]
            id1 = docset_id[-3]
            output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i}')
            with open(output_file, 'w', encoding='utf8') as outfile:
                for sentence in result:
                    print(sentence)
                    outfile.write(" ".join(sentence))
    else:
        for docset_id in tqdm(data, desc="Evaluating documents"):
            lx = LexRank.from_data(datafile=datafile, documentset=docset_id,
                sentences_are_documents=True, **args)
            if realization:
                _docset, indices = docset_loader(datafile, docset_id)
                result = lx.obtain_summary(_docset, coreference_resolution = True, detokenize=True)
            else:
                result = lx.obtain_summary(coreference_resolution = False, detokenize=True)
            result = create_clusters(docset_id, result, fractional_order, datafile)
            id0 = docset_id[0:5]
            id1 = docset_id[-3]
            output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i}')
            with open(output_file, 'w', encoding='utf8') as outfile:
                for sentence in result:
                    print(sentence)
                    outfile.write(" ".join(sentence))


if __name__ == '__main__':
    main()