'''
File contains module to select content via LexRank algorithm:
References:
    - Erkan, Günes, and Dragomir R. Radev. 
      "Lexrank: Graph-based lexical centrality as salience in text summarization." 
      Journal of artificial intelligence research 22 (2004): 457-479.
      https://www.jair.org/index.php/jair/article/view/10396
'''

import numpy as np
import pandas as pd
from vectorizer import DocumentToVectors, DocumentToVectorsFactory
from math import sqrt
from utils import detokenizer_wrapper
from nltk.metrics.distance import jaccard_distance
from typing import *
import logging
import argparse
import json, os
from tqdm import tqdm

Literal = List

logger = logging.getLogger()


def LexRankFactory(
        vector_generator: Literal['word2vec, tfidf, distilbert'],
        **kwargs
    ):
    '''
    Pass the __init__ function used to initiliaze a LexRank class to the
    data factory function
    Arguments:
        - vector_generator: whether to use 'word2vec', 'tfidf', 'distilbert' vectors
        - kwargs: additional arguments for the vector_generator (e.g., eval_documents for tfidf)
                  and arguments for lexrank (e.g., min_jaccard_dist)
    '''
    def LexRankInit(
            self,
            documents: List[List[str]],
            indices: Dict[str, Union[np.array, List[float]]],
            threshold: Optional[float] = 0.,
            error: Optional[float] = 1e-16,
            dampening_factor: Optional[float] = 0.15,
            min_length: Optional[int] = None, 
            min_jaccard_dist: Optional[float] = None,
            **kwargs
        ):
        '''
        threshold: the threshold for constructing graph connections
        error: the minimum error to end the power_method algorithm
        d: the dampening to ensure aperiodicity
        min_length is the minimum length (minimum number of tokens) as sentence needs tos have
        min_jaccard_dist will reject sentences that are below a certain jaccard distance 
            (a difference measure between sets)
        '''
        self.threshold = threshold
        self.error = error
        self.dampening_factor = dampening_factor
        self.min_length = min_length
        self.min_jaccard_dist = min_jaccard_dist
        self.raw_docs = documents

    return DocumentToVectorsFactory(
        LexRank, vector_generator, LexRankInit, **kwargs
    )


class LexRank(DocumentToVectors):
    '''Subclass with methods specific to LexRank
    Can only be initialized with DocumentToVectorsFactory
    '''

    def __init__(self, **kwargs) -> None:
        raise AttributeError("LexRank cannot be instantiated on its own.\n"+
                             "Use the LexRankFactory function to dynamically create " +
                             "a class and instantiated it with that new class.")

    def get_cosine_matrix(self) -> np.ndarray:
        '''
        Get the cosine matrix for the document
        Arguments:

        Returns:
            np.ndarray
        Citation: https://stackoverflow.com/q/21226610/
        '''
        matrix = self.similarity_matrix()
        matrix[matrix < self.threshold] = 0.
        return matrix


    def solve_lexrank(
            self, 
            return_type: Optional[Literal["'pandas', 'vector', 'list'"]] = 'pandas'
        ) -> Union[np.ndarray, pd.DataFrame]:
        '''
        Find the largest eigenvalue of the modified cosine matrix
        similarity matrix
        Arguments
            - return_type: whether to return one of the following options:
                a. the vector of eigenvalues
                b. a list of ranked of tuples (index, eigenvalue, sentence)
                c. (default) a pandas dataframe constructed from list of tuples
        Returns:
            A list, dataframe, or vector of the resulting eigenvalue
        '''
        if self.N > 1:
            eigenvalue = power_method(
            matrix=self.get_cosine_matrix(),
            error=self.error,
            d=self.dampening_factor
        )
        else: # if document only consists of one sentence
            eigenvalue = np.ones(shape=(1))
        if return_type == 'vector':
            return eigenvalue
        ranked_list = sorted(
            [(i, ev, self.raw_docs[i]) for i, ev in enumerate(eigenvalue.tolist())],
            key=lambda x: x[1],
            reverse=True
        )
        if return_type == 'list':
            ranked_list
        df = pd.DataFrame(ranked_list).reset_index()
        df.columns = ['rank', 'index', f'LR Score ({self.threshold})', 'sentence']
        return df


    @detokenizer_wrapper
    def obtain_summary(
            self, 
            max_tokens: Optional[int] = 100,
            detokenize: Optional[Union[Callable, bool]] = False
        ) -> Union[str, List[List[str]]]:
        '''
        Obtain a "summary" of the document by running the method `solve_lexrank`
        and then selecting sentences until it reaches the max number of words
        Arguments:
            - max_words: max tokens in the summary
            - detokenize: whether to combine the tokens into a typical English sentence
                or leave as a list of whitespace delimited tokens. The decorator 
                wrap_detokenizer transforms the tokenize bool into a function behind the scenes
        '''
        ranked_list = self.solve_lexrank('pandas')
        first_sentence = ranked_list['sentence'][0]
        words = len(first_sentence)
        if len(ranked_list['sentence']) == 1:
            if words < max_tokens:
                return detokenize(first_sentence)
            else:
                logger.warning(f"Highest ranked sentence has more than 100 tokens..." + \
                    "returning a slice of the sentence")
                return detokenize(first_sentence[0:max_tokens-1])
        else:
            i = 0
            summary_ids = []
            current_sentence = first_sentence
            while words < max_tokens and i + 1 < ranked_list.shape[0]:
                if self.min_jaccard_dist is not None:
                    too_similar = False
                    for previous_sent_id in summary_ids:
                        prev_sent = ranked_list['sentence'][previous_sent_id]
                        jaccard_d = jaccard_distance(set(current_sentence), set(prev_sent))
                        print(jaccard_d, self.min_jaccard_dist, too_similar, jaccard_d <= self.min_jaccard_dist)
                        if jaccard_d <= self.min_jaccard_dist:
                            too_similar = True
                            break
                if self.min_jaccard_dist is not None and too_similar:
                    i += 1
                    current_sentence = ranked_list['sentence'][i]
                    continue
                else:
                    summary_ids.append(i)
                    i += 1
                    current_sentence = ranked_list['sentence'][i]
                    words += len(current_sentence)
            summary = [detokenize(ranked_list['sentence'][sum_id]) for sum_id in summary_ids]
            assert words - len(current_sentence) < max_tokens, \
                f"words: {words - len(current_sentence)} | sentence: {ranked_list['sentence']}"
            if isinstance(summary[-1], str):
                summary = list(map(lambda x: x + "\n", summary))
            return detokenize(summary)


def power_method(
        matrix: np.ndarray, 
        error: float,
        d: float,
        max_iter: int = 10_000
    ) -> np.ndarray:
    '''
    Power method for solving stochastic, irreducible, aperiodic matrices
    Arguments:
        - matrix: a square matrix
        - error: when the error is low enough to finish algorithm
        - d: dampening factor (to ensure convergence)
    '''
    N = matrix.shape[0]
    p_t = np.ones(shape=(N))/N
    t, delta = 0, None
    U = np.ones(shape=matrix.shape)
    M = ((U * d)/N + (matrix * (1-d))/N).transpose()
    iterations = 0
    for i in range(N):
        if np.sum(M[:, i]) == 0.:
            M[:, i] = np.ones(shape=(N))/N
    while (delta is None or delta > error) and iterations < max_iter:
        t += 1
        p_t_1 = p_t.copy()
        p_t = M @ p_t
        delta = np.linalg.norm(p_t - p_t_1)
        iterations += 1
    # normalize ranking
    p_t = p_t/np.max(p_t)
    return p_t


def parse_args():
    argparser = argparse.ArgumentParser()
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
    args, _ = argparser.parse_known_args()
    return args


# def main():
#     args = parse_args()
#     assert args.data_set in set(['training', 'evaltest', 'devtest'])
#     fname = os.path.join(args.data_path, args.data_set + ".json")
#     with open(fname, 'r') as datafile:
#         data = json.load(datafile)
#     for docset_id in tqdm(data):
#         docset = data[docset_id]
#         lx = LexRank(docset, max_length=5, doc_level='sentence', 
#             punctuation=True, lowercase=True, min_jaccard_dist=0.6)
#         result = lx.obtain_summary(args.threshold, args.error, detokenize=True)
#         print(result)
#         # spl = str(docset_id).split("-", maxsplit=1)
#         # id0, id1 = spl[0], spl[1]
#         # output_file = os.path.join('outputs', 'D3', f'{id0}-A.M.100.{id1}.2')
#         # with open(output_file, 'w') as outfile:
#         #     outfile.write(result)


if __name__ == '__main__':
    pass
