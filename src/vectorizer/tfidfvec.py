from .vector_api import VectorModel, DocumentToVectors
from utils import CounterDict
from nltk.util import ngrams
from math import log
import re
from copy import deepcopy
from typing import *


# FOR TF-IDF, what you need to do is generate one hot vectors and create a matrix that way
# TODO: Add more specification details from tfidf


def create_idf_dictionary(documents: List[List[str]], delta_idf: float) -> Dict[str, float]:
    '''
    Create an inverse document frequency dictionary from a list of documents
    '''
    dictionary = CounterDict()
    for doc in documents:
        seen_words = set()
        for word in doc:
            if word not in seen_words:
                dictionary[word] += 1
    dictionary = dictionary.map(lambda x: log(len(documents)/(x + delta_idf)) + delta_idf)
    return dictionary


class TFIDFModel(VectorModel):
    def __init__(
            self, 
            documents: List[List[str]],
            ignore_punctuation: bool = True,
            lowercase: bool = True,
            ngram: int = 1,
            delta_idf: float = 0.7,
            log_tf: bool = False
        ) -> None:
        '''
        Instantiate a TFIDF dictionary from the provided data
        Args:
            - documents: The documents with which you build a TF-IDF matrix
            - ignore_punctuation: whether to include or eliminate punctuation
            - lowercase: whether to lowercase the words
            - ngram: whether to use ngram of 1, 2, or 3, default is 1
            - log_idf: whether to take the log of the IDF value or not
            - smoothing: whether to perform smoothing or not
            - delta_idf: IDF smoothing value
        '''
        docs = deepcopy(documents)
        self.ignore_punctuation, self.lowercase = ignore_punctuation, lowercase
        self.delta_idf = delta_idf
        self.log_tf = log_tf
        self.ngram = ngram
        process_docs = self._preprocess(docs)
        self.idf = create_idf_dictionary(process_docs, self.delta_idf)


    def _preprocess(self, docs):
        if self.ignore_punctuation:
            docs = [[w for w in doc if re.search(r'\w', w)] for doc in docs]
        if self.lowercase:
            docs = [[w.lower() for w in doc] for doc in docs]
        if self.ngram > 1:
            docs = [str(ngrams(doc, self.ngram)) for doc in docs]
        return docs


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
        tfidf = tf * self.idf if not self.log_tf else log(1 + tf) * self.idf
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
        if eval_documents:
            eval_docs = self._preprocess(eval_documents) 
        else:
            eval_docs = documents
        self.document_vectors = [self.vectorize_sentence(doc) for doc in eval_docs]
        self.N = len(eval_docs)
        self.indices = indices
    

if __name__ == '__main__':
    from utils import docset_loader
    docs = docset_loader('D1001A-A', 'data/devtest.json')
    x = DocumentToTFIDF(documents=docs)
    print(x.similarity_matrix())
