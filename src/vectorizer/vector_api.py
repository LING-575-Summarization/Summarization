from abc import ABC, abstractmethod
import numpy as np
from utils import docset_loader, dataset_loader
from pathlib import Path
from typing import *


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

    delattr(VectorModel, '__call__')

    def __init__(
            self, 
            documents: List[List[str]],
            indices: Dict[str, Union[np.array, List[float]]],
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
        super().__init__(**kwargs)
        self.document_vectors = [self.vectorize_sentence(doc) for doc in documents]
        self.indices = indices
        self.N = len(documents)


    def __getitem__(self, _key) -> Union[np.array, List[float]]:
        '''
        Retreive the vector of a single document
        If docs are docs:
            - _key should be {docset_id}.{doc_id} (e.g., D1001A-A.APW19990421.0284)
        If docs are sentences:
            - _key should be {docset_id}.{sentence_index} (e.g., D1001A-A.0)
        
        Args:
            - _key: either the index of the document or a string reference to the document
                    using docset id (e.g., D1001A-A), and document id (e.g., NYT...) or
                    sentence number (e.g, 2)
        '''
        if isinstance(_key, int) and _key < self.N:
            return self.document_vectors[_key]
        elif isinstance(_key, str):
            if _key in self.indices:
                index = self.indices[_key]
                return self.document_vectors[index]
            else:
                raise ValueError(
                    f"Can't find index for key {_key} " + 
                    f"(sample format: {list(self.indices.keys())[0]})"
                )
        else:
            raise TypeError(f"Unrecognized/Incompatible key type: {type(_key)}")


    def similarity_measure(self, i: int, j:int) -> float:
        '''
        Get the cosine similarity of documents using indices i and j
        Used by similarity_matrix method
        Args:
            - i: document index i 
            - j: document index j
        '''
        if i == j:
            return 0.
        else:
            v_i, v_j = self[i], self[j]
        if np.sum(v_i) == 0. or np.sum(v_j) == 0.:
            return 0.
        else:
            return np.dot(v_i, v_j)/(np.linalg.norm(v_i) * np.linalg.norm(v_j))


    def similarity_matrix(self, normalize: Optional[bool] = False) -> np.ndarray:
        '''
        Create a similarity matrix comparing the all documents in self.document_vectors
        
        Arguments:
            - normalize: Whether to normalize the rows in the matrix (can be helpful before
              applying a threshold to a sparse or low-value matrix)
        '''
        if self.document_vectors:
            matrix = np.array(
                [[
                    self.similarity_measure(i, j) if i!=j else 0. for j in range(self.N)
                    ] for i in range(self.N)
                ]
            )
            if normalize:
                matrix = np.apply_along_axis(
                    func1d=lambda x: x/x.sum() if x.sum() > 0 else x,
                    axis=0,
                    arr=matrix
                )
            return matrix
        else:
            raise AttributeError(f"Class {type(self)} not instantiated with document list")
        

    @classmethod
    def from_data(
        cls, 
        datafile: Path, 
        documentset: Optional[str] = None,
        eval_docset: Optional[str] = None,
        sentences_are_documents: Optional[bool] = False,
        **kwargs
    ):
        '''
        Load a DocumentToVectors class from a dataset
        Arguments:
            - datafile: Path to the datafile (e.g., data/devtest.json)
            - documentset: (Optional) The document set to analyze. 
              NOTE: if documentset is None, it will load the whole dataset, and disregard
                    the distinction between docsets
            - sentences_are_documents: whether to treat articles as documents (default) or
              treat sentences as documents
            - kwargs: arguments specific to the DocumentToVectors class (e.g., reduction for 
              word2vec)
        '''
        if documentset and eval_docset is None:
            documents, indices = docset_loader(datafile, documentset, sentences_are_documents)
        elif documentset is not None and eval_docset is not None:
            if "TFIDF" in cls.__name__:
                documents, _ = dataset_loader(datafile, sentences_are_documents)
                eval_documents, indices = docset_loader(
                    datafile, eval_docset, sentences_are_documents)
                kwargs['eval_documents'] = eval_documents
            else:
                raise TypeError("eval_docset is only a valid argument for DocumentToTFIDF")
        else:
            documents, indices = dataset_loader(datafile, sentences_are_documents)
        return cls(documents=documents, indices=indices, **kwargs)
    