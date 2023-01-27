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
from math import log, e, sqrt
from counterdict import CounterDict
from functools import reduce
from typing import *

# term frequency (tf):
    # return count(sentence, word_appears_in_sentence)

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
        body = document.pop()
        self.headers = document

        # get body as list of sentences
        def flatten(x,y): 
            x.extend(y)
            return x
        self.body = reduce(flatten, body)
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
        '''
        if s_i == s_j:
            return 0.
        else:
            sent_i_terms, sent_j_terms = set(self.body[s_i]), set(self.body[s_j])
            overlap_w = sent_i_terms.intersection(sent_j_terms)
            numerator = sum(
                [self.tf[s_i][w] * self.tf[s_i][w] * (self.idf[w] ** 2) for w in overlap_w]
            )
            denom_term = lambda s, s_t: sqrt(
                sum([(self.tf[s][x] * self.idf[x]) ** 2 for x in s_t])
            )
            denominator = denom_term(s_i, sent_i_terms) * denom_term(s_j, sent_j_terms)
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
            d: Optional[float] = 0.15
        ) -> np.ndarray:
        '''
        Find the largest eigenvalue of the modified cosine matrix
        similarity matrix
        Arguments:
            - threshold: the threshold for constructing graph connections
            - error: the minimum error to end the power_method algorithm
            - d: the dampening to ensure aperiodicity
        Returns:
            Vector (np.ndarray)
        '''
        return power_method(
            matrix=self.get_cosine_matrix(threshold=threshold),
            error=error,
            d=d
        )


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
    fname = os.path.join(os.path.dirname(__file__), 'testcase.json')
    with open(fname, 'r') as testfile:
        testcase = json.load(testfile)
    testcase = testcase["D1101A-A"]["AFP_ENG_20061002.0523"]
    lx = LexRank(testcase)
    eig = lx.solve_lexrank(0.1, 1e-8)
    x = sorted(
        [(" ".join(lx.body[i]), i, e) for i, e in enumerate(eig.tolist())],
        key=lambda x: x[2],
        reverse=True
    )
    print(x)