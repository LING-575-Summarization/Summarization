## Build an abstract class from constructing dictionaries, matrices, etc.
## The abstract methods for getting vectors are implemented downstream

from .bertvec import DistilBertModel, DocumentToDistilBert
from .word2vec import Word2VecModel, DocumentToWord2Vec
from .tfidfvec import TFIDFModel, DocumentToTFIDF
from .vector_api import DocumentToVectors
from .class_factory import DocumentToVectorsFactory

if __name__ == '__main__':
    pass