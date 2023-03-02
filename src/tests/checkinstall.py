'''
This file check that all NLTK and spaCy corpuses are installed.
Add your corpuses to this test file!
'''

import spacy
import nltk

def check_install():
    try:
        nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.corpus.stopwords.words()
    except LookupError:
        nltk.download('stopwords')
    try:
        spacy.load('en_core_web_md')
    except LookupError:
        spacy.download('en_core_web_md')
    try:
        spacy.load('en_core_web_sm')
    except LookupError:
        spacy.download('en_core_web_sm')
    return None