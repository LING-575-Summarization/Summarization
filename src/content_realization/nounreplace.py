'''
This module modifies extracted summaries so that the longest premodifying referent is placed at the
front of summaries while the remaining referents are replaced by the shortest non-pronoun one
'''

from typing import *
from .corefextract import ContentRealizer, DETOKENIZER
from nltk.tokenize import word_tokenize, sent_tokenize


def clean_up_realization(summary: str, max_tokens=100):
    print(summary, type(summary))
    retokenized_summary = [word_tokenize(s) for s in sent_tokenize(summary)]
    while len(retokenized_summary) > max_tokens:
        retokenized_summary.pop()
    sentences = [DETOKENIZER.detokenize(s) for s in retokenized_summary]
    capitalized = [s.capitalize() for s in sentences]
    return " ".join(capitalized)


def replace_referents(summary: str, original_documents: List[List[str]]) -> str:
    '''
    Replaces references in a summary with desired nouns
    '''
    cr = ContentRealizer()
    new_summary = cr(summary, original_documents)
    return clean_up_realization(new_summary)
    

if __name__ == '__main__':
    pass
