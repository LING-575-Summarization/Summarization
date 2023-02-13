from vector_api import VectorModel, DocumentToVectors
from transformers import DistilBertModel, DistilBertTokenizerFast
import numpy as np
from typing import *

'''
Citations:
- Google News Corpus (https://code.google.com/archive/p/word2vec/)
- Word2Vec paper (https://arxiv.org/pdf/1301.3781.pdf)
'''

class DistilBertModel(VectorModel):
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
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # set up the function used to obtain sentence vector

        if reduction == 'centroid':
            self.reduction_fn = lambda m: np.mean(np.vstack(m), axis=1)
        elif reduction == 'normalized_mean':
            self.reduction_fn = lambda m: np.mean(
                np.vstack([row/np.linalg.norm(row) for row in m]), 
                axis=1
            )
        elif reduction == 'normalized_sum':
            self.reduction_fn = lambda m: np.sum(
                np.vstack([row/np.linalg.norm(row) for row in m]), 
                axis=1
            )
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}")
        
    def vectorize_sentence(self, sentence: List[str]) -> List[float]:
        '''
        Return a vector representation of the sentece
        Also removes punctuation since punctuation is not accepted by word2vec
        Args:
            - sentence: a tokenized list of words
        Returns:
            - 1 dimensional np.ndarray of floats
        '''
        tokenized_sentence = self.tokenizer(sentence)
        vector = self.model()
        return self.reduction_fn(list_of_word_vectors)
    

class Word2VecToDocument(DocumentToVectors, Word2VecModel):
    
    def similarity_measure(self, i: int, j:int) -> float:
        '''
        Get the cosine similarity of documents using indices i and j
        Used by similarity_matrix method
        Args:
            - i: document index i 
            - j: document index j
        '''
        v_i, v_j = self.document_vectors[i], self.document_vectors[j]
        return np.dot(v_i, v_j)/(np.linalg.norm(v_i) * np.linalg.norm(v_j))
    

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
