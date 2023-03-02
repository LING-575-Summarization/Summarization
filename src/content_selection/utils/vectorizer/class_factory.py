import numpy as np
from inspect import signature
from typing import *

from .vector_api import DocumentToVectors
from .word2vec import DocumentToWord2Vec
from .tfidfvec import DocumentToTFIDF
from .bertvec import DocumentToDistilBert

def DocumentToVectorsFactory(
        subclass: DocumentToVectors,
        vector_generator: Literal['word2vec', 'tfidf', 'distilbert'],
        init_function: Callable,
        **kwargs
    ):

    if vector_generator == 'word2vec':
        subclass_generator = DocumentToWord2Vec
    elif vector_generator == 'tfidf':
        subclass_generator = DocumentToTFIDF
    elif vector_generator == 'distilbert' or vector_generator == 'bert':
        subclass_generator = DocumentToDistilBert
    else:
        raise ValueError(f"Unrecognized vector_generator: {vector_generator}")

    def __init__(
            self, 
            documents: List[List[str]],
            indices: Dict[str, Union[np.array, List[float]]],
            **kwargs
        ):
        init_fn_signature = signature(init_function)
        init_fn_kwargs = {}
        for parameter in init_fn_signature.parameters:
            if parameter in kwargs:
                init_fn_kwargs[parameter] = kwargs.pop(parameter)
        init_function(self, documents, indices, **init_fn_kwargs)
        subclass_generator.__init__(self, documents=documents, indices=indices, **kwargs)
    
    name = f"{subclass.__name__}{subclass_generator.__name__.split('To')[-1]}"
    bases = subclass_generator.__bases__
    dictionary = {**subclass_generator.__dict__, **subclass.__dict__}
    dictionary['__init__'] = __init__
    new_class = type(name, bases, dictionary, **kwargs)
    return new_class