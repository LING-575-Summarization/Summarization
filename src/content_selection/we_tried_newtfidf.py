'''
Document for new TF-IDF class
'''

from typing import *
from math import e, log
import re
import logging
import json
from utils import CounterDict, flatten_list
from nltk.util import ngrams

logger = logging.getLogger()
Literal = List


''' ####### Utility functions ####### '''


def process_docset(
        docset: List[List[List[str]]], 
        punctuation: bool,
        lowercase: bool
    ) -> List[List[Any]]: 
    '''
    Utility function to remove punctuation from a document set and
    put all terms to lowercase
    Note: flatten the List of Paragraphs for each document, so each document has [Sent[Token]] nesting
    Arguments:
        - docset: a list of documents which contain lists of tokenized sentences
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
        new_docset.append([casing(w) for w in doc_i if punctuation_filter(w)])
    return new_docset


''' ####### TF-IDF class ####### '''

class TFIDF:
    '''Get TF-IDF values from *just* one document with multiple sentences'''


    def __init__(
        self, 
        document_set: Dict[str, List[List[str]]],
        punctuation: bool,
        lowercase: bool,
        doc_level: Literal["sentence, document, docset"],
        docset_id: Optional[str] = None,
        ngram: int = 1,
        log_tf: Optional[bool] = False,
        log_idf: Optional[bool] = True,
        smoothing: Optional[bool] = True,
        delta_tf: float = 0.01,
        delta_idf: float = 0.01,
        log_base: Optional[Union[float, int]] = e,
        post_init: Optional[bool] = True
    ) -> None:
        '''
        Initialize a TF-IDF class to obtain two dictionaries:
            1. term frequency for each sentence
            2. inverse term frequency for each term
        Argument:
            - document_set: a dictionary (representing one docSet) that maps document ids
              to lists of tokenized sentences
            - punctuation: whether to include or eliminate punctuation
            - lowercase: whether to lowercase the words
            - doc_level: whether to consider passages or sentences as
              documents for TF-IDF calculations
            - docset_id: Needed if we want tf-score over docSet level, default is None
            - ngram: whether to use ngram of 1, 2, or 3, default is 1
            - log_ff: whether to take the log of the TF value or not
            - log_idf: whether to take the log of the IDF value or not
            - smoothing: whether to perform smoothing or not
            - delta_tf: TF smoothing value
            - delta_idf: IDF smoothing value
            - log_base: whether to use log base of 2 or e
            - TODO: post_init: ????
        '''
        # prepare documents to get attributes
        self.headers = [doc[0:-1] for doc in document_set.values()]
        raw_docs = [doc[-1] for doc in document_set.values()]
        
        # flatten paragraphs:
        self.raw_docs = [flatten_list(doc) for doc in raw_docs]
        self.doc_ids = list(document_set.keys())
        self.doc_level = doc_level
        self.docset_id = docset_id
        self.ngram = ngram
        self.punctuation = punctuation
        self.lowercase = lowercase
        self.log_tf = log_tf
        self.log_idf = log_idf
        self.log_base = log_base
        self.smoothing = smoothing
        self.tf = None
        self.idf = None

        # set deltas based on smoothing
        if self.smoothing:
            self.delta_tf, self.delta_idf = delta_tf, delta_idf
        else:
            self.delta_tf, self.delta_idf = 0., 0.

        self._set_up_tf_idf_functions()

        self._set_up_documents()

        self.N = len(self.docs)

        # checks
        if not(self.log_idf) and self.log_base != e:
            logger.warning(
                f"log_idf is False but self.log_base is specified. Ignoring self.log_base ({self.log_base})..."
            )
        if not(self.smoothing) and (self.delta_tf != 0 or self.delta_idf != 0):
            logger.warning(
                f"smoothing is False but self.delta_tf or self.delta_idf is specified. Ignoring smoothing..."
            )

        if post_init:
            self.__post_init__()


    def _set_up_tf_idf_functions(self):
        '''Set up the tf and idf functions based on smoothing and delta'''
        # tf and idf functions
        if self.smoothing:
            _function_tf = lambda x: x + self.delta_tf
            _function_idf = lambda x: x + self.delta_idf
        else:
            _function_tf = lambda x: x
            _function_idf = lambda x: x
        if self.log_tf:
            function_tf = lambda x: log(_function_tf(x), self.log_base)
        else:
            function_tf = lambda x: _function_tf(x)
        if self.log_idf:
            function_idf = lambda x: log(_function_idf(x), self.log_base) + self.delta_idf
        else:
            function_idf = lambda x: _function_idf(x)

        self.function_tf, self.function_idf = function_tf, function_idf


    # set up docs for `for loop`
    def _set_up_documents(self):
        if (self.ngram > 3) or (self.ngram < 1):
            raise ValueError(
                f"ngram argument must be 1, 2, or 3, not {self.ngram}"
            )

        if self.doc_level == 'sentence':
            # turn each document set into a list of sentences
            new_doc_ids = []
            for doc_id, doc in zip(self.doc_ids, self.raw_docs):
                new_doc_ids.extend([doc_id + f".{i+1}" for i in range(len(doc))])
            self.doc_ids = new_doc_ids
            self.raw_docs = flatten_list(self.raw_docs)
            documents = process_docset(self.raw_docs, self.punctuation, self.lowercase)
            self.docs = documents
        elif self.doc_level == 'document':
            # turn each document into a list of tokens
            self.raw_docs = [flatten_list(doc) for doc in self.raw_docs]
            documents = process_docset(
                self.raw_docs, self.punctuation, self.lowercase
            )
            self.docs = documents
        elif self.doc_level == "docset":
            # turn the given docset into a list of tokens, wrapped in a list
            self.raw_docs = [flatten_list(flatten_list(self.raw_docs))]
            documents = process_docset(
                self.raw_docs, self.punctuation, self.lowercase
            )
            if self.docset_id is None:
                raise ValueError(
                    f"docset_id argument must be a str, not default None for docset level only"
                )
            self.doc_ids = [self.docset_id]
            self.docs = documents
        else:
            raise ValueError(
                f"doc_level argument must be either sentence, document, or docset, not {self.doc_level}"
            )


    def __post_init__(self):
        '''
        Get the tf and idf dictionaries from the specified document set
            (apply after __init__, but as a separate call)
        '''
        tf, df = {}, CounterDict()
        for id, doc in zip(self.doc_ids, self.docs):
            tf_doc = CounterDict()
            seen_words = set()
            for word in doc:
                tf_doc[word] += 1
                if word not in seen_words:
                    seen_words.add(word)
                    df[word] += 1
            tf[id] = tf_doc
        idf = df.map(lambda x: self.N/x)
        self.tf, self.idf = self._smoothing_and_log(tf, idf)

    
    def _smoothing_and_log(self, tf: Dict[str, Any], idf: CounterDict):
        '''
        Applies smoothing and log transformations as defined by the class
        '''
        for doc_i, tf_doc in tf.items():
            tf[doc_i] = tf_doc.map(self.function_tf)

        idf = idf.map(self.function_idf)

        return tf, idf


    def get(self, word: str, document_id: str, return_doc: Optional[bool] = False):
        """
            Arguments:
                - word: which ngram you want the tfidf score for
                - document_id: which document this ngram comes from
                - return_doc: verbose output, optional default is False
        """
        term_freq, inv_doc_freq = self.tf[document_id][word], self.idf[word]
        # print(term_freq, "(", sum([1 if self.tf[d][word]>=1 else 0 for d in self.doc_ids]), self.N, ")", inv_doc_freq)
        
        if term_freq == 0 and self.smoothing:
           term_freq = self.function_tf(0)
        if inv_doc_freq == 0 and self.smoothing:
            inv_doc_freq = self.function_idf(0)

        if return_doc:
            index = self.doc_ids.index(document_id)
            return term_freq * inv_doc_freq, self.docs[index]
        else:
            return term_freq * inv_doc_freq


    def __getitem__(self, doc_and_word: Tuple[str]):
        word, doc_id = doc_and_word
        return self.get(word, doc_id)


    def _tf_count(self) -> Tuple[List[Dict[str, int]], Dict[str, int]]:
        '''
        Count just term frequencies
        '''
        tf = {}
        for i, doc in enumerate(self.docs):
            tf_doc = CounterDict()
            for word in doc:
                tf_doc[word] += 1
            tf[self.doc_ids[i]] = tf_doc
        return tf


    def _idf_count(self) -> Tuple[List[Dict[str, int]], Dict[str, int]]:
        '''
        Count just document frequencies
        '''
        df = CounterDict()
        for doc in self.docs:
            seen_words = set()
            for word in doc:
                if word not in seen_words:
                    seen_words.add(word)
                    df[word] += 1
        return df


    @classmethod
    def idf_from_docset(
            cls,
            tf_documents: Dict[str, List[List[str]]],
            idf_documents: Dict[str, List[List[str]]],
            punctuation: bool,
            lowercase: bool,
            doc_level: Literal["sentence, document, docset"],
            docset_id: Optional[str] = None,
            **kwargs,
        ) -> None:
        '''
            Create a TF-IDF dictionary using separate document sets for
            TF and IDF
            Use with TFIDF.idf_from_docset(...)
        '''
        tfidf = cls(document_set=tf_documents, punctuation=punctuation, lowercase=lowercase, doc_level=doc_level, docset_id=docset_id, post_init=True, **kwargs)
        _idf = cls(document_set=idf_documents, punctuation=punctuation, lowercase=lowercase, doc_level=doc_level, docset_id=docset_id, post_init=False, **kwargs)

        tf, idf = tfidf._tf_count(), _idf._idf_count()


        print("old idf", tfidf.idf)

        tfidf.tf, idf2 = tfidf._smoothing_and_log(tf, idf)
        print("new idf", idf2)

        return tfidf


    def __str__(self):
        _d = self.__dict__
        d = _d.copy()
        d['doc_ids'] = list(set(_d['doc_ids']))
        for key, val in _d.items():
            if key in ['headers', 'raw_docs', 'docs', 'tf', 'idf']:
                if isinstance(val, list):
                    d[key] = f"list of {type(val[0])}"
                elif isinstance(val, CounterDict):
                    d[key] = f"CounterDict ({key})"
                elif isinstance(val, dict):
                    d[key] = f"dict ({key})"
                else:
                    d[key] = f"{type(val)}"
            else:
                d[key] = f"{type(val)}"
        return "Class: TFIDF " + json.dumps(d, indent=4)
        

if __name__ == '__main__':
    # with open('/home2/hsteinm/575-Summarization/data/devtest.json', 'r') as infile:
    #     docset_rep = json.load(infile)
    # docset = docset_rep['D1001A-A']
    # print(docset)
    # tfidf = TFIDF(
    #     docset,
    #     punctuation=True,
    #     lowercase=True,
    #     doc_level='docset',
    #     log_tf=False,
    #     log_idf=True,
    #     smoothing=True
    # )
    # print(tfidf)
    # val, string = tfidf.get('columbine', 'NYT19990424.0231.1', True)
    # print(val, string)
    # print("df", string.count('columbine'))

    # del tfidf
    # tfidf = TFIDF(docset, True, True, 'sentence')
    # val, string = tfidf.get('columbine', 'NYT19990424.0231.1', True)
    # print("tf", string)

    #############################
    # for Sam's testing
    with open('output/data/devtest.json', 'r') as infile:
        docset_rep = json.load(infile)
    tf_docset_id = 'D1001A-A'
    tf_docset = docset_rep[tf_docset_id]

    idf_docset = {}
    for docset_id, docset in docset_rep.items():
        new_key = docset_id

        for document, data in docset.items():
            new_key += "." + document
            idf_docset[new_key] = data

    # tfidf = TFIDF(
    #     tf_docset,
    #     punctuation=True,
    #     lowercase=True,
    #     doc_level='docset',
    #     docset_id = tf_docset_id,
    #     log_tf=False,
    #     log_idf=True,
    #     smoothing=True
    # )

    # print(tf_docset)
    tfidf = TFIDF.idf_from_docset(tf_docset, idf_docset, True, True, "docset", docset_id=tf_docset_id)

    # print(tfidf.idf)

    # print(tfidf)
    val, string = tfidf.get('a', tf_docset_id, True)
    # print(val, string)
    # val = tfidf.get('columbine', docset_id, False)
    print(val)
    # print("df", string.count('columbine'))
