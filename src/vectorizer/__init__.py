## Build an abstract class from constructing dictionaries, matrices, etc.
## The abstract methods for getting vectors are implemented downstream

from .bertvec import DistilBertModel, DocumentToDistilBert
from .word2vec import Word2VecModel, DocumentToWord2Vec
from .tfidfvec import TFIDFModel, DocumentToTFIDF
from .vector_api import DocumentToVectors

import numpy as np
from typing import *

def DocumentToVectorsFactory(
        subclass: DocumentToVectors,
        vector_generator: Literal['word2vec', 'tfidf', 'distilbert'],
        **kwargs
    ):

    if vector_generator == 'word2vec':
        subclass_generator = DocumentToDistilBert
    elif vector_generator == 'tfidf':
        subclass_generator = DocumentToWord2Vec
    elif vector_generator == 'distilbert' or vector_generator == 'bert':
        subclass_generator = DocumentToTFIDF
    else:
        raise ValueError(f"Unrecognized vector_generator: {vector_generator}")

    def __init__(
            self, 
            documents: List[List[str]],
            indices: Dict[str, Union[np.array, List[float]]],
            max_length: Optional[int] = None, 
            min_jaccard_dist: Optional[float] = None, 
            **kwargs
        ):
        '''
        max_length is the maximum length (maximum number of tokens) as sentence can have
        min_jaccard_dist will reject sentences that are below a certain jaccard distance 
            (a difference measure between sets)
        '''
        self.max_length = max_length
        self.min_jaccard_dist = min_jaccard_dist
        self.raw_docs = documents
        subclass_generator.__init__(self, documents=documents, indices=indices, **kwargs)
    
    name = f"{subclass.__name__}{subclass_generator.__name__.split('To')[-1]}"
    bases = subclass_generator.__bases__
    dictionary = {**subclass_generator.__dict__, **subclass.__dict__}
    dictionary['__init__'] = __init__
    new_class = type(name, bases, dictionary, **kwargs)
    return new_class

if __name__ == '__main__':
    pass