'''
Document for new TF-IDF class
'''

from typing import *
from math import e, log
import re
import logging
from utils import CounterDict, flatten_list

logger = logging.getLogger()
Literal = List


''' ####### Utility function ####### '''


def process_body(
        docset: List[List[List[str]]], 
        punctuation: bool,
        lowercase: bool
    ) -> List[List[Any]]: 
    '''
    Utility function to remove punctuation from a body and
    put all terms to lowercase
    Arguments:
        - docset: a of documents which contain lists of tokenized sentences
        - punctuation: whether to remove punctuation or not from the sentences
        - lowercase: whether to lowercase words in the tokenized sentences
    '''
    if punctuation:
        punctuation_filter = lambda w: True if re.search(r'\w', w) else False
    else:
        punctuation_filter = lambda w: True
    if lowercase:
        casing = lambda w: w.lower()
    else:
        casing = lambda w: w
    new_docset = []
    for doc_i in docset:
        for sentence_i in range(len(doc_i)):
            new_docset.append([casing(w) for w in docset[sentence_i] if punctuation_filter(w)])
    return new_docset


''' ####### TF-IDF class ####### '''

class TFIDF:
    '''Get TF-IDF values from *just* one document with multiple sentences'''

    def __init__(
            self, 
            document: List[List[List[str]]], 
            punctuation: bool,
            lowercase: bool,
            doc_level: Literal["sentence", "document"],
            log_tf: Optional[bool] = False,
            log_idf: Optional[bool] = False,
            smoothing: Optional[bool] = True,
            delta_tf: float = 1.,
            delta_idf: float = 1.,
            log_base: Optional[Union[float, int]] = e,
        ) -> None:
        '''
        Initialize a TF-IDF class to obtain a two dictionaries: 
            1. term frequency for each sentence
            2. inverse term frequency for each term
        Argument:
            - document: sentences stored in a list of lists and 
              sentences are separated by paragraphs
            - punctuation: whether to include or eliminate punctuation
            - level: whether to consider passages or sentences as
              documents for TF-IDF calculations
            - log_ff: whether to take the log of the TF value or not
            - log_idf: whether to take the log of the IDF value or not
            - smoothing: whether to perform smoothing or not
            - delta_tf: TF smoothing value
            - delta_idf: IDF smoothing value
            - log_base: whether to use log base of 2 or e
        '''

        self.headers = []
        self.raw_docs = [doc[-1] for doc in document]
        self.docs = process_body(self.raw_docs, punctuation, lowercase)
        self.doc_level = doc_level
        self.log_tf = log_tf
        self.log_idf = log_idf
        self.log_base = log_base
        self.smoothing = smoothing
        if self.smoothing:
            self.delta_tf, self.delta_idf = delta_tf, delta_idf
        else:
            self.delta_tf, self.delta_idf = 0., 0.

        # checks            
        if not(self.log_idf) and self.log_base != e:
            logger.warning(
                f"log_idf is False but self.log_base is specified. Ignoring self.log_base ({self.log_base})..."
            )
        if not(self.smoothing) and (self.delta_tf != 1 or self.delta_idf != 1):
            logger.warning(
                f"smoothing is False but self.delta_tf or self.delta_idf is specified. Ignoring smoothing..."
            )
        if self.doc_level not in ['sentence', 'document']:
            raise ValueError(
                f"doc_level argument must be either sentence or document, not {self.doc_level}"
            )


    def __post_init__(self):
        '''
        Get the tf and idf dictionaries from the specified document set
            (apply after __init__, but as a separate call)
        '''
        self.tf, self.idf = self._freq_counter(self.log_base)


    def _sentence_freq_counter(self) -> Tuple[List[Dict[str, int]], Dict[str, int]]:
        '''
        Driver for __post_init__
        '''
        
        tf, df = _sentence_counter(self.docs)

        return self._tf_idf_modifier(tf, df)


    def _freq_counter(self) -> Tuple[List[Dict[str, int]], Dict[str, int]]:
        '''
        Driver for __post_init__
        '''
        self.docs = flatten_list(self.docs)
        self.N = len(self.docs)

        tf, df = [], CounterDict()
        for document in self.docs:
            tf_doc = CounterDict()
            list_of_tf_sent, df_sent = _sentence_counter(document)
            df.update_from_wordset(df_sent.keys())
            for tf_sent in list_of_tf_sent:
                tf_doc.update(tf_sent)
            tf.append(tf_doc)

        return self._tf_idf_modifier(tf, df)


    def _tf_idf_modifier(
            self, 
            tf: List[CounterDict], 
            idf: CounterDict
        ) -> Tuple[List[CounterDict], CounterDict]:
        '''
        Takes a list of term frequency counts and inverse document frequencies
        and modifies them based on class specifications (e.g. applies log function
        or delta corrections) 
        Part of the driver for __init__
        '''
        if self.log_tf:
            for tf_doc in tf:
                tf_doc.map(
                    lambda x: log(x + self.delta_tf, self.log_base)
                )

        idf.map(lambda x: self.N/x)
        if self.log_idf:
            idf.map(
                lambda x: log(self.N/x + self.delta_idf, self.log_base) + self.delta_idf
            )

        return tf, idf


    @classmethod
    def idf_from_docset(
            cls,
            tf_documents: List[List[List[str]]],
            idf_documents: List[List[List[str]]], 
            punctuation: bool,
            lowercase: bool,
            doc_level: Literal["sentence", "document"],
            **kwargs,
        ) -> None:
        '''
            Create a TF-IDF dictionary using separate document sets for
            TF and IDF
        '''
        tfidf = cls.__init__(tf_documents, punctuation, lowercase, doc_level, **kwargs)
        tf, idf = [], CounterDict()

        if tfidf.doc_level == 'sentence':
            # turn each document set into a list of sentences
            idf_documents = flatten_list(idf_documents)
        else:
            # turn each document into a list of tokens
            idf_documents = [flatten_list(idf_doc) for idf_doc in idf_documents]
        for idf_doc_i in idf_documents:
            idf_documents =  