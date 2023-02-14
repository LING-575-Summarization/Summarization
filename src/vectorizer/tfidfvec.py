from .vector_api import VectorModel, DocumentToVectors
from utils import CounterDict
import numpy as np
from math import log
import re
from copy import deepcopy
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
    dictionary = dictionary.map(lambda x: log(len(documents)/x))
    return dictionary


class TFIDFModel(VectorModel):
    def __init__(
            self, 
            documents: List[List[str]],
            ignore_punctuation: bool = True,
            lowercase: bool = True
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
        self.ignore_punctuation, self.lowercase = ignore_punctuation, lowercase

        if self.ignore_punctuation:
            docs = [[w for w in doc if re.search(r'\w', w)] for doc in docs]
        if self.lowercase:
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
        tf = CounterDict(keys=list(self.idf.keys()))
        for _word in sentence:
            if self.ignore_punctuation and re.search(r'\w', _word) is None:
                continue
            word = _word.lower() if self.lowercase else _word
            if word in tf:
                tf[word] += 1
        tfidf = tf * self.idf
        tfidf_vector = tfidf.to_numpy()
        return tfidf_vector
    

class DocumentToTFIDF(DocumentToVectors, TFIDFModel):
    def __init__(
            self, 
            documents: List[List[str]], 
            indices: Dict[str, int],
            eval_documents: Optional[List[List[str]]] = None, 
            **kwargs
        ) -> None:
        '''
        Override metaclass __init__ method since DocumentToTFIDF takes additional arguments
        '''
        TFIDFModel.__init__(self, documents, **kwargs)
        eval_docs = eval_documents if eval_documents else documents
        self.document_vectors = [self.vectorize_sentence(doc) for doc in eval_docs]
        self.N = len(eval_docs)
        self.indices = indices
    

if __name__ == '__main__':
    from utils import docset_loader
    docs = docset_loader('D1001A-A', 'data/devtest.json')
    x = DocumentToTFIDF(documents=docs)
    print(x.similarity_matrix())
