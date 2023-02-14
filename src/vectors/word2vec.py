from vector_api import VectorModel, DocumentToVectors
import gensim
from gensim.models import FastText, KeyedVectors
import gensim.downloader as api
import os
import numpy as np
import re
from typing import *

'''
Citations:
- Google News Corpus (https://code.google.com/archive/p/word2vec/)
- Word2Vec paper (https://arxiv.org/pdf/1301.3781.pdf)
'''

class Word2VecModel(VectorModel):
    def __init__(
            self, 
            reduction: Literal['centroid', 'normalized_mean', 'normalized_sum'] = 'centroid'
        ) -> None:
        '''
        Instantiate a word2vec dictionary from the Google News Corpus and select
        a reduction method to turn a sentence into a single vector (default is centroid)
        Also instantiates a Fasttext model to account for unseen vocabulary items
        Args:
            - reduction: Takes a list of word vectors (numpy arrays) and obtains
              a sentence-level representational vector
                - centroid: average all the word vectors in the sentence
                - normalized_mean: normalize the vectors with L2 norm then
                  average all the word vectors in the sentence 
                - normalized_sum: normalize the vectors with L2 norm then
                  sum all the word vectors in the sentence 
        '''

        self.vector_size = 300
        self.model = api.load('word2vec-google-news-300')

        # set up the function used to obtain sentence vector
        if reduction == 'centroid':
            self.reduction_fn = lambda m: np.mean(np.vstack(m), axis=0)
        elif reduction == 'normalized_mean':
            self.reduction_fn = lambda m: np.mean(
                np.vstack([row/np.linalg.norm(row) for row in m]), 
                axis=0
            )
        elif reduction == 'normalized_sum':
            self.reduction_fn = lambda m: np.sum(
                np.vstack([row/np.linalg.norm(row) for row in m]), 
                axis=0
            )
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}")
        
    def vectorize_sentence(self, sentence: List[str]) -> np.ndarray:
        '''
        Return a vector representation of the sentece
        Also removes punctuation since punctuation is not accepted by word2vec
        Args:
            - sentence: a tokenized list of words
        Returns:
            - 1 dimensional np.ndarray of floats
        '''
        list_of_word_vectors = []
        for word in sentence:
            # if punctuation
            if not re.search(r'\w', word):
                continue
            # if in vocabulary
            if word in self.model:
                vec = self.model[word]
            list_of_word_vectors.append(vec)
        return self.reduction_fn(list_of_word_vectors)
    

class Word2VecToDocument(DocumentToVectors, Word2VecModel):
    pass
    

if __name__ == '__main__':
    import json
    from functools import reduce

    # get body as list of sentences
    def flatten_list(x: List[List[Any]]) -> List[Any]: 
        '''
        Utility function to flatten lists of lists
        '''
        def flatten(x, y):
            x.extend(y)
            return x
        return reduce(flatten, x)


    with open('data/devtest.json', 'r') as datafile:
        data = json.load(datafile)
    data = data['D1001A-A']
    docs = [flatten_list(d) for d in [flatten_list(doc[-1]) for doc in data.values()]]
    x = Word2VecToDocument(documents=docs, reduction='centroid')
    print(x.similarity_matrix())
