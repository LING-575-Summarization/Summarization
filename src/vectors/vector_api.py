from abc import ABC, abstractmethod
from inspect import signature
import numpy as np
from typing import *


# TODO: Function to parse raw documents


class VectorModel(ABC):

    @abstractmethod
    def vectorize_sentence(self) -> np.ndarray:
        '''
        Method implemented by subclass to create a vector representation of the sentece
        '''
        pass


    def __call__(self, sentence: List[str]) -> np.ndarray:
        return self.vectorize_sentence(sentence)


class DocumentToVectors(VectorModel):

    def __init__(
            self, 
            documents: List[List[str]],
            **kwargs
        ) -> None:
        '''
        Use a model to create vectors for an entire set of documents 
        Args:
            - documents: the documents to process. must be a list of lists of strings. the
              desired document-level (e.g., sentence-level vs. document-level) must be set
              by the user. this class does not automatically infer the level. so, if the 
              list of whole documents is provided, the class will calculate a vector to 
              represent the WHOLE document and not the sentences within the document 
            - kwargs: arguments specific to the implementation of DocumentToVectors
              e.g., reduction for word2vec implementation
        '''
        # check if the subclass (i.e. the _Model class) needs documents to be initialized
        # this check handles the tfidf vectorizer
        super_signature = signature(super().__init__)
        if 'documents' in super_signature.parameters:
            kwargs['documents'] = documents
        
        super().__init__(**kwargs)
        self.document_vectors = [self.vectorize_sentence(doc) for doc in documents]
        self.N = len(documents)


    def __call__(self, documents: List[List[str]]) -> List[np.ndarray]:
        return [self.vectorize_sentence(doc) for doc in documents]


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


    def similarity_matrix(self) -> np.ndarray:
        '''
        Create a similarity matrix comparing the all documents in self.document_vectors
        '''
        if self.document_vectors:
            matrix = np.array(
                [[self.similarity_measure(i, j) for j in range(self.N)] for i in range(self.N)]
            )
            return matrix
        else:
            raise AttributeError(f"Class {type(self)} not instantiated with document list")