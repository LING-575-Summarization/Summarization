'''
This file check that all NLTK and spaCy corpuses are installed.
Add your corpuses to this test file!
'''

import spacy
import nltk

def check_install():
    try:
        nltk.data.load('tokenizers/punkt/english.pickle')
    except (LookupError, IOError) as error:
        nltk.download('punkt')
    try:
        nltk.corpus.stopwords.words()
    except (LookupError, IOError) as error:
        nltk.download('stopwords')
    try:
        spacy.load('en_core_web_md')
    except (LookupError, IOError) as error:
        spacy.cli.download('en_core_web_md')
    try:
        spacy.load('en_core_web_sm')
    except (LookupError, IOError) as error:
        spacy.cli.download('en_core_web_sm')
    try:
        t = nltk.treebank.parsed_sents('wsj_0001.mrg')
    except (LookupError, IOError) as error:
        spacy.cli.download('en_core_web_sm')
    return None