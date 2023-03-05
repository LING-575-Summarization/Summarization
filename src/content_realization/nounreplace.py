'''
This module modifies extracted summaries so that the longest premodifying referent is placed at the
front of summaries while the remaining referents are replaced by the shortest non-pronoun one
'''

from typing import *
from .corefextract import ContentRealizer, DETOKENIZER
from nltk.tokenize import word_tokenize
from functools import reduce
import re


def clean_up_realization(summary: str, max_tokens=100):
    retokenized_summary = [word_tokenize(s) for s in summary]
    if len(retokenized_summary) > 1:
        num_tokens = lambda z: reduce(
            lambda x, y: x + len(y) if isinstance(x, int) else len(x)+ len(y), 
            z)
        while len(retokenized_summary) > 1 and num_tokens(retokenized_summary) > max_tokens:
            retokenized_summary.pop()
    if len(retokenized_summary) == 1 and len(retokenized_summary[0]) > max_tokens:
        retokenized_summary[0] = retokenized_summary[0][0:100]
    summary_string = " ".join([DETOKENIZER.detokenize(s) for s in retokenized_summary])
    summary_string = re.sub(r'\s([,\.\"\'?!%])(?=\s)', '\1', summary_string)
    summary_string = re.sub(r'(?<=\w)([,\.?!%])(?=\w+)', '\1 ', summary_string)
    return summary_string


def replace_referents(summary: str, original_documents: List[List[str]]) -> Tuple[str, bool]:
    '''
    Replaces references in a summary with desired nouns
    '''
    cr = ContentRealizer()
    new_summary, success_in_replace = cr(summary, original_documents)
    return clean_up_realization(new_summary), success_in_replace
    

if __name__ == '__main__':
    pass
