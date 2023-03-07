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
        spacy.load('en_core_web_trf')
    except (LookupError, IOError) as error:
        import subprocess
        subprocess.run(
            ["pip", "install", 
             "https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl"
            ])

    return None

if __name__ == '__main__':
    check_install()