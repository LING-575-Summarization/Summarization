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
from utils import CounterDict, detokenizer_wrapper
from functools import reduce
from typing import Optional, Union, List, Literal, Tuple, Dict, Callable, Any
import logging

logger = logging.getLogger()

# get body as list of sentences
def flatten_list(x: List[List[Any]]) -> List[Any]: 
    '''
    Utility function to flatten lists of lists
    '''
    def flatten(x, y):
        x.extend(y)
        return x
    return reduce(flatten, x)


def process_body(body: List[List[str]]) -> List[List[Any]]: 
    '''
    Utility function to remove punctuation from a body and
    put all terms to lowercase
    '''
    new_body = []
    for sentence_i in range(len(body)):
        new_body.append([w.lower() for w in body[sentence_i] if re.search(r'\w', w)])
    return new_body


class TFIDF:
    '''Get TF-IDF values from *just* one document with multiple sentences'''

    def __init__(
            self, 
            document: List[List[List[str]]], 
            log_base: Optional[Union[float, int]] = e
        ) -> None:
        '''
        Obtain a two dictionaries: 
            1. term frequency for each sentence
            2. inverse term frequency for each term
        Argument:
            - document: sentences stored in a list of lists and 
              sentences are separated by paragraphs
        '''
        body = document[-1]
        self.headers = document

        self.raw_body = flatten_list(body)
        self.body = process_body(self.raw_body)
        self.N = len(self.body)

        # checks
        assert all([isinstance(sent, list) for sent in self.body]), "Not all of body are sentences"
        assert all([all([isinstance(tkn, str) for tkn in sent]) for sent in self.body]), "Not all of body are sentences"
        
        self.tf, self.idf = self._document_counter(log_base)


    def _document_counter(
            self,
            log_base: Union[float, int]
        ) -> Tuple[List[Dict[str, int]], Dict[str, int]]:
        '''
        Driver for __init__
        '''
        tf, df = [], CounterDict()

        for sentence in self.body:
            seen_words = set()
            tf_sentence = CounterDict()
            for word in sentence:
                word = word.lower()
                if re.search(r'\w+', word) is None: # avoid punctuation
                    pass
                else:
                    tf_sentence[word] += 1
                    if word not in seen_words:
                        df[word] += 1
                        seen_words.add(word)
            tf.append(tf_sentence)

        idf = df.map(
            lambda x: log(self.N/x, log_base)
        )

        return tf, idf


class LexRank(TFIDF):
    '''Subclass with methods specific to LexRank'''

    def __init__(
            self, 
            document: List[List[List[str]]], 
            log_base: Optional[Union[float, int]] = e
        ) -> None:
        super().__init__(document, log_base)


    def modified_cosine(self, s_i: int, s_j: int) -> float:
        '''Helper method to get the modified cosine score specific in Erkan and Radev
            Arguments:
                - s_i, s_j: indices to the sentences in self.body and self.tf
            NOTE: Self links (i = j) are allowed
        '''
        sent_i_terms, sent_j_terms = set(self.body[s_i]), set(self.body[s_j])
        one_sentence_has_no_terms = len(sent_i_terms) == 0 or len(sent_j_terms) == 0
        if one_sentence_has_no_terms: # i.e. sentence is entirely punctuation
            return 0.
        else:
            overlap_w = sent_i_terms.intersection(sent_j_terms)
            numerator = sum(
                [self.tf[s_i][w] * self.tf[s_i][w] * (self.idf[w] ** 2) for w in overlap_w]
            )
            denom_term = lambda s, s_t: sqrt(
                sum([(self.tf[s][x] * self.idf[x]) ** 2 for x in s_t])
            )
            denominator = denom_term(s_i, sent_i_terms) * denom_term(s_j, sent_j_terms)
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
            return_type: Optional[Literal['pandas', 'vector', 'list']] = 'pandas'
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
        if len(self.body) > 1:
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
            [(i, ev, self.raw_body[i]) for i, ev in enumerate(eigenvalue.tolist())],
            key=lambda x: x[1],
            reverse=True
        )
        if return_type == 'list':
            ranked_list
        df = pd.DataFrame(ranked_list).reset_index()
        df.columns = ['rank', 'index', f'LR Score ({threshold})', 'sentence']
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
            if words <= max_tokens:
                return detokenize(first_sentence)
            else:
                logger.warning(f"Highest ranked sentence has more than 100 tokens..." + \
                    "returning a slice of the sentence")
                return detokenize(first_sentence[0:100])
        else:
            summary, current_sentence, i = [], first_sentence, 1
            while words < max_tokens and i < ranked_list.shape[0]:
                summary.append(detokenize(current_sentence))
                current_sentence = ranked_list['sentence'][i]
                i += 1
                words += len(current_sentence)
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


# NOTE: want the method to be able to handle both one document
#   and multiple documents. in each case, you just tokenize the document 
#   into sentences and run the algorithm like usual

if __name__ == '__main__':
    import json, os
    from tqdm import tqdm
    fname = os.path.join('data', 'training.json')
    with open(fname, 'r') as testfile:
        training_data = json.load(testfile)
    with open('output/train-lex-rank.txt', 'w') as outfile:
        with tqdm(training_data, leave=False, total=len(training_data) * 10) as pbar:
            for docset in pbar:
                for doc_id in training_data[docset]:
                    lx = LexRank(training_data[docset][doc_id])
                    result = lx.obtain_summary(0.1, 1e-8, detokenize=True)
                    print(result, file=outfile)
                    pbar.update(1)
                    if isinstance(result, str):
                        from nltk.tokenize import word_tokenize
                        assert len(word_tokenize(result)) < 100, f"{result}"
                    else:
                        assert len(flatten_list(result)) < 100, f"{result}"