from vector_api import VectorModel, DocumentToVectors
from counterdict import CounterDict
from math import log
import re
from copy import deepcopy
import numpy as np
from typing import *


# FOR TF-IDF, what you need to do is generate one hot vectors and create a matrix that way


def create_idf_dictionary(documents: List[List[str]]) -> Dict[str, float]:
    '''
    Create an inverse document frequency dictionary from a list of documents
    '''
    dictionary = CounterDict()
    for doc in documents:
        seen_words = set()
        for word in doc:
            if word not in seen_words:
                dictionary[word] += 1
    dictionary = dictionary.map(lambda x: log(len(documents)/(x + 1)) + 1)
    return dictionary


class TFIDFModel(VectorModel):
    def __init__(
            self, 
            documents: List[List[str]],
            ignore_punctuation: bool = True,
            lowercase: bool = True,
            reduction: Literal['centroid', 'normalized_mean', 'normalized_sum'] = 'centroid'
        ) -> None:
        '''
        Instantiate a TFIDF dictionary from the Google News Corpus and select
        a reduction method to turn a sentence into a single vector (default is centroid)
        Also instantiates a Fasttext model to account for unseen vocabulary items
        This implementation uses log IDF scores with smoothing
        Args:
            - documents: The documents with which you build a TF-IDF matrix
            - reduction: Takes a list of word vectors (numpy arrays) and obtains
              a sentence-level representational vector
                - centroid: average all the word vectors in the sentence
                - normalized_mean: normalize the vectors with L2 norm then
                  average all the word vectors in the sentence 
                - normalized_sum: normalize the vectors with L2 norm then
                  sum all the word vectors in the sentence 
        '''
        docs = deepcopy(documents)
        if ignore_punctuation:
            docs = [[w for w in doc if re.search(r'\w', w)] for doc in docs]
        if lowercase:
            docs = [[w.lower() for w in doc] for doc in docs]

        self.idf = create_idf_dictionary(docs)


    def vectorize_sentence(self, sentence: List[str]) -> Dict[str, float]:
        '''
        Return a vector representation of the sentece using TFIDF
        Args:
            - sentence: a tokenized list of words
        Returns:
            - dictionary that maps words to tfidf values
        '''
        idf = self.idf if self.idf is not None else create_idf_dictionary(sentence)
        tf = CounterDict(keys=list(idf.keys()))
        for word in sentence:
            if word in tf:
                tf[word] += 1
        tfidf = tf * idf
        tfidf_vector = tfidf.to_numpy()
        return tfidf_vector
    

class TFIDFDocument(DocumentToVectors, TFIDFModel):
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
    x = TFIDFDocument(documents=docs, reduction='centroid')
    print(x.similarity_matrix())
