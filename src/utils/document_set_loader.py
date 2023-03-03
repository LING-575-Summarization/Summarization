import json
from utils import flatten_list
from collections import OrderedDict
from typing import *
from pathlib import Path


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
        if sentences_are_documents:
            doc_as_sentences = [flatten_list(d) for d in document[-1]]
            for j in range(len(doc_as_sentences)):
                indexes[doc_id + "." + str(j)] = doc_as_sentences[j]
            documents.extend(doc_as_sentences)
        else:
            document_sentences = [flatten_list(d) for d in document[-1]]
            if merge_sentences_to_doc:
                document_sentences = flatten_list(document_sentences)
            documents.append(document_sentences)
            indexes[doc_id] = i
    return documents, indexes
    

def dataset_loader(
        datafile: Path, 
        sentences_are_documents: Optional[bool] = False
    ) -> Tuple[List[List[str]], Dict[str, int]]:
    '''
    Read the whole data file as a list of documents. Docsets are ignored. This function
    is useful for calculating TFIDF on a training data
    Args:
        - datafile: The datafile to extract the document set from
        - sentences_are_documents: Whether to consider sentences as documents. If True,
          the function returns a list of sentences for all docs in the docset. If False,
          the function returns a list of tokenized documents for all docs in the docset.
    Returns:
        - Tuple: a list of all documents in the dataset and indices to reference those documents
    '''
    with open(datafile, 'r') as dfilestream:
        data = json.load(dfilestream)
    alldocuments, indexes = [], OrderedDict()
    for docset_id, docset in data.items():
        for i, (doc_id, document) in enumerate(docset.items()):
            if sentences_are_documents:
                doc_as_sentences = [flatten_list(d) for d in document[-1]]
                for j in range(len(doc_as_sentences)):
                    index_key = docset_id + "." + str(j)
                    indexes[index_key] = doc_as_sentences[j]
                alldocuments.extend(doc_as_sentences)
            else:
                alldocuments.append(flatten_list([flatten_list(d) for d in document[-1]]))
                index_key = docset_id + "." + doc_id
                indexes[index_key] = i
    return alldocuments, indexes
    