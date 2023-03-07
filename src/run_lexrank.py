from vectorizer import *
from tqdm import tqdm
import json, os
from utils import docset_loader
from extraction_methods.lexrank import LexRankFactory
from extraction_methods import SentenceIndex, create_clusters
from argparse import ArgumentParser
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()
detokenize = lambda x: detokenizer.detokenize(x)

def get_args():
    argparser = ArgumentParser()
    argparser.add_argument(
        '--data_path', '-dpath', type=str, default='../data',
        help='Path to document set directory'
    )
    argparser.add_argument(
        '--data_set', '-ds', type=str,
        help='Whether to use the test, dev, or train data split'
    )
    argparser.add_argument(
        '--threshold', '-t', type=float, default=0.15,
        help='The threshold to use when creating weighted graph'
    )
    argparser.add_argument(
        '--error', '-e', type=float, default=1e-10,
        help='The error to use when solving for the eigenvalue'
    )
    argparser.add_argument(
        "--min_length", type=int, default=7,
    )
    argparser.add_argument(
        "--vector", type=str, default='tfidf',
    )
    argparser.add_argument(
        "--min_jaccard_dist", type=float, default=0.6,
    )
    argparser.add_argument(
        "--content_realization", action='store_true'
    )
    argparser.add_argument(
        "--idf_docset", action='store_true'
    )
    argparser.add_argument(
        "--ngram", type=int, default=1,
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
        "--ignore_stopwords", action='store_true'
    )
    args, _ = argparser.parse_known_args()
    return args
    

def main():
    args = get_args()
    datafile = os.path.join(str(args.data_path), str(args.data_set) + ".json")
    fractional_order = SentenceIndex(datafile)
    with open(datafile, 'r') as evalfile:
        data = json.load(evalfile).keys()

    l_args = vars(args)
    l_args.pop('data_path')
    dataset_split = l_args.pop('data_set')

    vector_type = l_args.pop('vector')
    realization = l_args.pop('content_realization')
    
    LexRank = LexRankFactory(vector_type)

    idf_docset = l_args.pop('idf_docset')

    if idf_docset:

        lx = LexRank.from_data(datafile=[datafile, 'data/training.json'], sentences_are_documents=True,
                                do_evaluate=False, **l_args)
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
            output_file = os.path.join('outputs', f'D5_{dataset_split}', f'{id0}-A.M.100.{id1}.2')
            with open(output_file, 'w', encoding='utf8') as outfile:
                for sentence in result:
                    outfile.write(detokenize(sentence))
                    outfile.write("\n")

    else:

        for docset_id in tqdm(data, desc="Evaluating documents"):
            lx = LexRank.from_data(datafile=datafile, documentset=docset_id,
                sentences_are_documents=True, **l_args)
            if realization:
                _docset, indices = docset_loader(datafile, docset_id)
                result = lx.obtain_summary(_docset, coreference_resolution = True, detokenize=True)
            else:
                result = lx.obtain_summary(coreference_resolution = False, detokenize=True)
            result = create_clusters(docset_id, result, fractional_order, datafile)
            id0 = docset_id[0:5]
            id1 = docset_id[-3]
            output_file = os.path.join('outputs', f'D5_{dataset_split}', f'{id0}-A.M.100.{id1}.2')
            with open(output_file, 'w', encoding='utf8') as outfile:
                for sentence in result:
                    outfile.write(detokenize(sentence))
                    outfile.write("\n")


if __name__ == '__main__':
    main()