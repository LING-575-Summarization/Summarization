from .tfidfvec import DocumentToTFIDF
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np
from typing import *
    

class DocumentToLSA(DocumentToTFIDF):
    def __init__(
            self, 
            documents: List[List[str]], 
            indices: Dict[str, int],
            eval_documents: Optional[List[List[str]]] = None, 
            do_evaluate: bool = True,
            ignore_punctuation: bool = True,
            ignore_stopwords: bool = False,
            lowercase: bool = True,
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
        super().__init__(documents, indices, eval_documents, do_evaluate, 
                         ignore_punctuation=ignore_punctuation, ignore_stopwords=ignore_stopwords,
                         lowercase=lowercase, ngram=1, log_tf=False, delta_idf=0.)


    def fit_lsa(self):
        '''
        Fit the sentence vectors to the LSA
        '''
        document_array = np.stack(self.document_vectors)
        document_matrix = np. self.svd.fit(document_array)
        self.document_vectors = np.vsplit(self.svd.transform(document_matrix))
    

if __name__ == '__main__':
    from utils import docset_loader
    docs = docset_loader('D1001A-A', 'data/devtest.json')
    x = DocumentToTFIDF(documents=docs)
    print(x.similarity_matrix())
