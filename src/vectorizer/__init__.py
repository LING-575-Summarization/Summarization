## Build an abstract class from constructing dictionaries, matrices, etc.
## The abstract methods for getting vectors are implemented downstream

from .bertvec import DistilBertModel, DocumentToDistilBert
from .word2vec import Word2VecModel, DocumentToWord2Vec
from .tfidfvec import TFIDFModel, DocumentToTFIDF

if __name__ == '__main__':
    pass