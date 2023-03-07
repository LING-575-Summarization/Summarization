import json
from .processing_utils import flatten_list
from collections import OrderedDict
from typing import *
from pathlib import Path
import os
import re

def get_summaries(directory: Path, docset_id: Optional[str] = None):
    eval_files = OrderedDict()
    pattern = re.compile(f'{docset_id}\.\w') if docset_id else r'D10\d\d-A\.M\.100\.\w\.\w'
    for filename in os.listdir(directory):
        if re.search(pattern, filename):
            with open(os.path.join(directory, filename), 'r', encoding='cp1252') as summary:
                eval_files[filename] = summary.read()
    return eval_files


def docset_loader(
        datafile: Path, 
        docset: str, 
        sentences_are_documents: Optional[bool] = False,
        merge_sentences_to_doc: Optional[bool] = False
    ) -> Tuple[List[List[str]], Dict[str, int]]:
    '''
    Read a data file and obtain the document set as a list of tokenized strings
    Args:
        - docset: The document set to extract
        - datafile: The datafile to extract the document set from
        - sentences_are_documents: Whether to consider sentences as documents. If True,
          the function returns a list of sentences for all docs in the docset. If False,
          the function returns a list of tokenized documents for all docs in the docset.
        - merge_sentences_to_doc: Whether to merge the sentences in a doc into a single string
    Returns:
        - Tuple: a list of all documents in the dataset and indices to reference those documents
    '''
    with open(datafile, 'r') as dfilestream:
        data = json.load(dfilestream)
    data = data[docset]
    documents, indexes = [], OrderedDict()
    for i, (doc_id, document) in enumerate(data.items()):
        sentences = []
        for sent in document[-1]:
            if isinstance(sent[0], list):
                sentences.extend(sent)
            else:
                sentences.append(sent)
        if sentences_are_documents:
            for j in range(len(sentences)):
                indexes[doc_id + "." + str(j)] = sentences[j]
            documents.extend(sentences)
        else:
            documents.append(flatten_list(sentences))
            indexes[doc_id] = i
    return documents, indexes
    

def dataset_loader(
        datafile: Path, 
        sentences_are_documents: Optional[bool] = False,
        return_dict: Optional[bool] = False
    ) -> Tuple[List[List[str]], Dict[str, int]]:
    '''
    Read the whole data file as a list of documents. Docsets are ignored. This function
    is useful for calculating TFIDF on a training data
    Args:
        - datafile: The datafile to extract the document set from
        - sentences_are_documents: Whether to consider sentences as documents. If True,
          the function returns a list of sentences for all docs in the docset. If False,
          the function returns a list of tokenized documents for all docs in the docset.
        - sent_tokenize: Whether to return the documents as a dictionary of docsets to documents
    Returns:
        - Tuple: a list of all documents in the dataset and indices to reference those documents
    '''
    with open(datafile, 'r') as dfilestream:
        data = json.load(dfilestream)
    if return_dict:
        alldocuments = OrderedDict()
        for docset_id, docset in data.items():
            for doc_id, document in docset.items():
                sentences = []
                for sent in document[-1]:
                    if isinstance(sent[0], list):
                        sentences.extend(sent)
                    else:
                        sentences.append(sent)
                if sentences_are_documents:
                    for j, d in enumerate(sentences):
                        index_key = docset_id + "." + str(j)
                        alldocuments[index_key] = d
                else:
                    index_key = docset_id + "." + doc_id
                    alldocuments[index_key] = flatten_list(sentences)
        indexes = None
    else:
        alldocuments, indexes = [], OrderedDict()
        for docset_id, docset in data.items():
            for i, (doc_id, document) in enumerate(docset.items()):
                sentences = []
                for sent in document[-1]:
                    if isinstance(sent[0], list):
                        sentences.extend(sent)
                    else:
                        sentences.append(sent)
                if sentences_are_documents:
                    for j in range(len(sentences)):
                        index_key = docset_id + "." + str(j)
                        indexes[index_key] = len(alldocuments) + j
                    alldocuments.extend(sentences)
                else:
                    alldocuments.append(flatten_list(sentences))
                    index_key = docset_id + "." + doc_id
                    indexes[index_key] = i
    return alldocuments, indexes
    