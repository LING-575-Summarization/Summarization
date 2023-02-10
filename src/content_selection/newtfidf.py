'''
Document for new TF-IDF class
'''

from typing import *
from math import e, log
import re
import logging
import json
from utils import CounterDict, flatten_list

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
        doc_level: Literal["sentence, document"],
        log_tf: Optional[bool] = False,
        log_idf: Optional[bool] = True,
        smoothing: Optional[bool] = True,
        delta_tf: float = 0.01,
        delta_idf: float = 0.01,
        log_base: Optional[Union[float, int]] = e,
        post_init: Optional[bool] = True
    ) -> None:
        '''
        Initialize a TF-IDF class to obtain a two dictionaries: 
            1. term frequency for each sentence
            2. inverse term frequency for each term
        Argument:
            - document_set: a dictionary with the docset name mapping
              to a dictionary that maps document ids to lists of tokenized 
              sentences
            - punctuation: whether to include or eliminate punctuation
            - lowercase: whether to lowercase the words
            - level: whether to consider passages or sentences as
              documents for TF-IDF calculations
            - log_ff: whether to take the log of the TF value or not
            - log_idf: whether to take the log of the IDF value or not
            - smoothing: whether to perform smoothing or not
            - delta_tf: TF smoothing value
            - delta_idf: IDF smoothing value
            - log_base: whether to use log base of 2 or e
        '''

        # prepare documents to get attributes
        self.headers = [doc[0:-1] for doc in document_set.values()]
        raw_docs = [doc[-1] for doc in document_set.values()]
        
        # flatten paragraphs:
        self.raw_docs = [flatten_list(doc) for doc in raw_docs]
        self.doc_ids = list(document_set.keys())
        self.doc_level = doc_level
        self.punctuation = punctuation
        self.lowercase = lowercase
        self.log_tf = log_tf
        self.log_idf = log_idf
        self.log_base = log_base
        self.smoothing = smoothing

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
            documents = process_docset(
                [flatten_list(doc) for doc in self.raw_docs], self.punctuation, self.lowercase
            )
            self.docs = documents
        else:
            raise ValueError(
                f"doc_level argument must be either sentence or document, not {self.doc_level}"
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
        term_freq, inv_doc_freq = self.tf[document_id][word], self.idf[word]
        print(term_freq, "(", sum([1 if self.tf[d][word]>=1 else 0 for d in self.doc_ids]), self.N, ")", inv_doc_freq)
        
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
            tf_documents: List[List[List[str]]],
            idf_documents: List[List[List[str]]], 
            punctuation: bool,
            lowercase: bool,
            doc_level: Literal["sentence, document"],
            **kwargs,
        ) -> None:
        '''
            Create a TF-IDF dictionary using separate document sets for
            TF and IDF
            Use with TFIDF.idf_from_docset(...)
        '''
        tfidf = cls.__init__(tf_documents, punctuation, lowercase, doc_level, post_init=False, **kwargs)
        _idf = cls.__init__(idf_documents, punctuation, lowercase, doc_level, post_init=False, **kwargs)

        tf, idf = tfidf._tf_count(), _idf.idf_count()

        tfidf.tf, tfidf.idf = tfidf._smoothing_and_log(tf, idf)

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
    with open('/home2/hsteinm/575-Summarization/data/devtest.json', 'r') as infile:
        docset_rep = json.load(infile)
    docset = docset_rep['D1001A-A']
    print(docset)
    tfidf = TFIDF(
        docset, 
        punctuation=True, 
        lowercase=True, 
        doc_level='sentence', 
        log_tf=False, 
        log_idf=True, 
        smoothing=True
    )
    print(tfidf)
    val, string = tfidf.get('columbine', 'NYT19990424.0231.1', True)
    print(val, string)
    print("df", string.count('columbine'))
    # del tfidf
    # tfidf = TFIDF(docset, True, True, 'sentence')
    # val, string = tfidf.get('columbine', 'NYT19990424.0231.1', True)
    # print("tf", string)