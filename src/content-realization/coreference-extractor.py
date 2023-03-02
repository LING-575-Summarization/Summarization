'''
This file contains a class that performs the coreference resolution

From https://github.com/huggingface/neuralcoref:

    Attr or method	    Return type	    Description
    i                   int             Index of the cluster in the Doc
    main	            Span	        Span of the most representative mention in the cluster
    mentions	        list of Span	List of all the mentions in the cluster
    __getitem__	        return Span	    Access a mention in the cluster
    __iter__	        yields Span	    Iterate over mentions in the cluster
    __len__	return      int	            Number of mentions in the cluster
'''

import spacy
nlp = spacy.load('en_core_web_md')