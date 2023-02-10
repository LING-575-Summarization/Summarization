'''
File contains module to select content via LexRank algorithm:
References:
    - Erkan, Günes, and Dragomir R. Radev. 
      "Lexrank: Graph-based lexical centrality as salience in text summarization." 
      Journal of artificial intelligence research 22 (2004): 457-479.
      https://www.jair.org/index.php/jair/article/view/10396
'''

import re
import numpy as np
import pandas as pd
from math import log, e, sqrt
from newtfidf import TFIDF
from utils import CounterDict, detokenizer_wrapper, flatten_list
from typing import Optional, Union, List, Tuple, Dict, Callable, Any
import logging
import argparse
import json, os
from tqdm import tqdm

Literal = List

logger = logging.getLogger()


class LexRank(TFIDF):
    '''Subclass with methods specific to LexRank'''

    def __init__(self, doc_set, max_length = None, **kwargs):
        self.max_length = max_length
        super().__init__(document_set=doc_set, **kwargs)

    def modified_cosine(self, s_i: int, s_j: int) -> float:
        '''Helper method to get the modified cosine score specific in Erkan and Radev
            Arguments:
                - s_i, s_j: indices to the sentences in self.docs and self.tf
            NOTE: Self links (i = j) are allowed
        '''
        sent_i_terms, sent_j_terms = set(self.docs[s_i]), set(self.docs[s_j])
        one_sentence_has_no_terms = len(sent_i_terms) == 0 or len(sent_j_terms) == 0
        if one_sentence_has_no_terms: # i.e. sentence is entirely punctuation
            return 0.
        else:
            if self.max_length:
                too_few_terms = len(sent_i_terms) <= self.max_length or len(sent_j_terms) <= self.max_length
                if too_few_terms:
                    return 0.
            overlap_w = sent_i_terms.intersection(sent_j_terms)
            s_i_r, s_j_r = self.doc_ids[s_i], self.doc_ids[s_j]
            numerator = sum(
                [self.tf[s_i_r][w] * self.tf[s_i_r][w] * (self.idf[w] ** 2) for w in overlap_w]
            )
            denom_term = lambda s, s_t: sqrt(
                sum([(self.tf[s][x] * self.idf[x]) ** 2 for x in s_t])
            )
            denominator = denom_term(s_i_r, sent_i_terms) * denom_term(s_j_r, sent_j_terms)
            assert denominator != 0, f"Denominator of modified cosine is 0. Terms: {sent_i_terms}, {sent_j_terms}"
            return numerator/denominator


    def get_cosine_matrix(self, threshold: float) -> np.ndarray:
        '''
        Get the cosine matrix for the document
        Arguments:

        Returns:
            np.ndarray
        Citation: https://stackoverflow.com/q/21226610/
        '''
        f = lambda i, j: self.modified_cosine(i, j)
        matrix = np.array(
            [[f(i, j) for j in range(self.N)] for i in range(self.N)]
        )
        matrix[matrix < threshold] = 0.
        # normalize row sums
        matrix = np.apply_along_axis(
            func1d=lambda x: x/x.sum() if x.sum() > 0 else x,
            axis=0,
            arr=matrix
        )
        # compute 
        return matrix


    def solve_lexrank(
            self, 
            threshold: float, 
            error: float,
            d: Optional[float] = 0.15,
            return_type: Optional[Literal["'pandas', 'vector', 'list'"]] = 'pandas'
        ) -> Union[np.ndarray, pd.DataFrame]:
        '''
        Find the largest eigenvalue of the modified cosine matrix
        similarity matrix
        Arguments:
            - threshold: the threshold for constructing graph connections
            - error: the minimum error to end the power_method algorithm
            - d: the dampening to ensure aperiodicity
            - return_type: whether to return one of the following options:
                a. the vector of eigenvalues
                b. a list of ranked of tuples (index, eigenvalue, sentence)
                c. (default) a pandas dataframe constructed from list of tuples
        Returns:
            A list, dataframe, or vector of the resulting eigenvalue
        '''
        if len(self.docs) > 1:
            eigenvalue = power_method(
            matrix=self.get_cosine_matrix(threshold=threshold),
            error=error,
            d=d
        )
        else: # if document only consists of one sentence
            eigenvalue = np.ones(shape=(1))
        if return_type == 'vector':
            return eigenvalue
        ranked_list = sorted(
            [(i, ev, self.raw_docs[i], self.docs[i]) for i, ev in enumerate(eigenvalue.tolist())],
            key=lambda x: x[1],
            reverse=True
        )
        if return_type == 'list':
            ranked_list
        df = pd.DataFrame(ranked_list).reset_index()
        df.columns = ['rank', 'index', f'LR Score ({threshold})', 'sentence', 's']
        return df


    @detokenizer_wrapper
    def obtain_summary(
            self, 
            threshold: float, 
            error: float,
            d: Optional[float] = 0.15,
            max_tokens: Optional[int] = 100,
            detokenize: Optional[Union[Callable, bool]] = False
        ) -> Union[str, List[List[str]]]:
        '''
        Obtain a "summary" of the document by running the method `solve_lexrank`
        and then selecting sentences until it reaches the max number of words
        Arguments:
            - threshold: the threshold for constructing graph connections
            - error: the minimum error to end the power_method algorithm
            - d: the dampening to ensure aperiodicity
            - max_words: max tokens in the summary
            - detokenize: whether to combine the tokens into a typical English sentence
                or leave as a list of whitespace delimited tokens. The decorator 
                wrap_detokenizer transforms the tokenize bool into a function behind the scenes
        '''
        ranked_list = self.solve_lexrank(threshold, error, d, 'pandas')
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
            summary, current_sentence, i = [], first_sentence, 1
            while words < max_tokens and i < ranked_list.shape[0]:
                summary.append(detokenize(current_sentence))
                current_sentence = ranked_list['sentence'][i]
                i += 1
                words += len(current_sentence)
            assert words - len(current_sentence) < max_tokens, \
                f"words: {words - len(current_sentence)} | sentence: {ranked_list['sentence']}"
            if isinstance(summary[-1], str):
                summary = list(map(lambda x: x + "\n", summary))
            return detokenize(summary)


def power_method(
        matrix: np.ndarray, 
        error: float,
        d: float
    ) -> np.ndarray:
    '''
    Power method for solving stochastic, irreducible, aperiodic matrices
    Arguments:
        - matrix: a square matrix
        - error: when the error is low enough to finish algorithm
        - d: dampening factor (to ensure convergence)
    '''
    p_t = np.ones(shape=(matrix.shape[0]))/matrix.shape[0]
    t, delta = 0, None
    U = np.ones(shape=matrix.shape)/matrix.shape[0]
    while delta is None or delta > error:
        t += 1
        p_t_1 = p_t
        p_t = np.matmul(
            (U * d) + (matrix.T * (1-d)), 
            p_t
        )
        delta = np.linalg.norm(p_t - p_t_1)
    # normalize ranking
    p_t = p_t/p_t.sum()
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


def main():
    args = parse_args()
    assert args.data_set in set(['training', 'evaltest', 'devtest'])
    fname = os.path.join(args.data_path, args.data_set + ".json")
    with open(fname, 'r') as datafile:
        data = json.load(datafile)
    for docset_id in tqdm(data):
        docset = data[docset_id]
        lx = LexRank(docset, max_length=5, doc_level='sentence', punctuation=True, lowercase=True)
        result = lx.obtain_summary(args.threshold, args.error, detokenize=True)
        print(result)
        # spl = str(docset_id).split("-", maxsplit=1)
        # id0, id1 = spl[0], spl[1]
        # output_file = os.path.join('outputs', 'D3', f'{id0}-A.M.100.{id1}.2')
        # with open(output_file, 'w') as outfile:
        #     outfile.write(result)


if __name__ == '__main__':
    main()
