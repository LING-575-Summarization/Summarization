from vector_api import VectorModel, DocumentToVectors
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
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
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased",
            add_special_tokens=True
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            output_hidden_states=True
        )
        self.max_length = self.model.config.max_length

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
        Also truncates sentences that are too long
        Args:
            - sentence: a tokenized list of words
        Returns:
            - 1 dimensional np.ndarray of floats
        '''
        sentence_as_string = " ".join(sentence)
        print(len(sentence))
        tokenized_sentence = self.tokenizer(
            sentence_as_string, return_tensors='pt', max_length=self.max_length
        )
        with torch.no_grad():
            hidden_states = self.model(**tokenized_sentence).hidden_states
        cls_token = hidden_states[0].squeeze()[:, 0]
        return cls_token
    

class DistilBertToDocument(DocumentToVectors, DistilBertModel):
    
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
    x = DistilBertToDocument(documents=docs, reduction='centroid')
    print(x.similarity_matrix())