from .tfidfvec import DocumentToTFIDF
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np
from typing import *
    

class DocumentToLSA(DocumentToTFIDF):
    def __init__(
            self, 
            *args,
            **kwargs
        ) -> None:
        '''
        Override metaclass __init__ method since DocumentToTFIDF takes additional arguments
        '''
        self.n_components = kwargs.pop('n_components', 300)
        self.svd = TruncatedSVD(self.n_components)
        if any([param in ['ngram', 'delta_idf', 'log_tf'] for param in kwargs]):
            print("Warning: LSA does not accept 'ngram', 'delta_idf', 'log_tf' as arguments."
                  "Removing these keyword arguments.")
        kwargs['ngram'] = 1
        kwargs['log_tf'] = 1
        kwargs['delta_idf'] = 1
        DocumentToTFIDF.__init__(self, *args, **kwargs)
        self.fit_lsa()


    def fit_lsa(self):
        '''
        Fit the sentence vectors to the LSA
        '''
        document_matrix = np.stack(self.document_vectors)
        self.svd = self.svd.fit(document_matrix)
        document_vectors = np.vsplit(self.svd.transform(document_matrix), 
                                          document_matrix.shape[0])
        self.document_vectors = [v.squeeze() for v in document_vectors]
        assert np.all([v.shape == self.document_vectors[0].shape for v in self.document_vectors])
    

if __name__ == '__main__':
    from utils import docset_loader
    docs = docset_loader('D1001A-A', 'data/devtest.json')
    x = DocumentToTFIDF(documents=docs)
    print(x.similarity_matrix())
