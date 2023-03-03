'''
This file contains a class that performs the coreference resolution
The implementation here is inspired by Nenkova (2007)
'''

from typing import *
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils import detokenize_list_of_tokens
from copy import deepcopy
from tqdm import tqdm
import re

import warnings
warnings.filterwarnings('ignore', 
                        message='User provided device_type of \'cuda\', but CUDA is not available. Disabling', 
                        category=Warning)

DETOKENIZER = TreebankWordDetokenizer()


class Span:
    '''Ad-hoc dataclass to store information on spans in a way that's compatible with spacy'''
    def __init__(self, tokens: List["token"]) -> None:
        self.tokens = tokens
        self.start = tokens[0].i
        self.end = tokens[-1].i
        
    @property
    def text(self):
        return DETOKENIZER.detokenize([tkn.text for tkn in self.tokens])

    def __iter__(self):
        return iter(self.tokens)
    
    def __str__(self):
        return self.text
    
    def __len__(self):
        return len(self.tokens)


class ReferenceClusters:

    def __init__(
            self, 
            doc: str, 
            cluster: str,
            pronoun_tags = {"PRP", "PRON", "PRP$"},
            determiner_tags = {"DT", "WDT", "DET"}
        ) -> None:
        '''Class to hold coreference solutions and extract longest/shortest values'''
        self.doc = doc
        self.cluster = cluster
        self.pronoun_tags = pronoun_tags
        self.determiner_tags = determiner_tags
        self.spans = self.post_init()

    def __iter__(self):
        return iter(self.doc.spans[self.cluster])

    def post_init(self):
        '''Remove any postpositions and filter pronouns'''
        filtered, filtered_appositives = [], []
        for span in self.doc.spans[self.cluster]:
            tags = [tkn.tag_ for tkn in span]
            if not("," in tags or "$," in tags):
                filtered_appositives.append(span)
            else:
                tag_indices = [i for i, t in enumerate(tags) if t == "," or t == "$,"]
                tokens = self.doc[span[0].i:span[tag_indices[0]].i]
                new_span = Span(tokens)
                filtered_appositives.append(new_span)
        for span in filtered_appositives:
            if not(len(span) == 1 or contains_a_pronoun(span, self.pronoun_tags)):
                filtered.append(span)
        return filtered

    @property
    def _spans(self):
        return self.doc.spans[self.cluster]

    @property
    def _lengths(self):
        return [len(sp.text) for sp in self.doc.spans[self.cluster]]
    
    @property
    def lengths(self):
        return [len(sp.text) for sp in self.spans]

    @property
    def shortest(self):
        lengths = self.lengths
        if len(lengths) > 0:
            min_length = min(lengths)
            argmin = lengths.index(min_length)
            return self.spans[argmin]
        else:
            return None

    def shortest_nonproper(self):
        min_length = min(self._lengths)
        argmin = self.lengths.index(min_length)
        return self.doc.spans[self.cluster][argmin]

    @property
    def longest(self):
        lengths = self.lengths
        # if more than one value
        if len(lengths) > 0:
            max_length = max(lengths)
            argmax = lengths.index(max_length)
            longest = self.spans[argmax]
            return longest
        else:
            return None



def recover_document(sentence: List[str], document_set: List[List[str]]) -> Tuple[int, int]:
    '''
    Iterate through a document set to recover the original indices of the sentences
    The document set should be a list of a list of sentences. Can use docset_loader to reload
        sentences if you'd like
    If there are repeated sentences across documents, pick the first one.
    '''
    for i, doc in enumerate(document_set):
        for j, s in enumerate(doc.sents):
            if sentence == s.text:
                return (doc, j)
    return None


def contains_a_pronoun(span, pronoun_tags: Set[str] = {"PRP", "PRON", "PRP$"}):
    return len(set([tkn.tag_ for tkn in span]) & pronoun_tags) > 0


def replace_via_indices(string: str, sub: str, start: int, end: int) -> str:
    '''Helper function to replace a string with another one at specified indices'''
    return "".join([string[:start], sub, string[end:]])


def replace_nth(string: str, sub: str, replacement: str, n: int):
    '''Replaces the string with the longest/shortest entity
       NOTE: It also corrects for replacements that use "I" in quotes. 
             When it is followed by a "said." string
    '''
    pattern = re.compile(f'(?<=[\s\"\']){sub}(?=[\s\"\'\.,?!])')
    split = re.split(pattern, string)
    start, end = f"{sub}".join(split[0:n]), f"{sub}".join(split[n:])
    is_quote = start.count("\"") % 2 == 1 and end.count("\"") % 2 == 1
    replacing_I = sub == "I" and re.search(f".*said", string) is not None
    if is_quote and replacing_I:
        return string, False
    elif is_quote and not replacing_I:
        replacement = f"[{replacement}]"
    return start + replacement + end, True


class CoferenceResolver:
    nlp = spacy.load("en_coreference_web_trf")
    parser = spacy.load("en_core_web_md")

    def __call__(self, in_doc: str) -> Any:
        '''
        Return the parsed document
        '''
        return self.nlp(self.parser(in_doc))

    
    def prettyprint(self, input: str):
        doc = self.nlp(self.parser(input))
        for i, sp in enumerate(doc.spans.values()):
            print("Group", i+1)
            for span in sp:
                print("\t", span.text)
                for tkn in span:
                    print("\t\t", tkn.text, tkn.tag_)


class ContentRealizer:
    resolver = CoferenceResolver()

    def __call__(self, summary: List[List[str]], 
                 document_set: List[List[str]]) -> None:
        # Sorry if the function is long... SpaCy's garbage collection is pretty aggresive towards 
        # out-of-scope variables, so I kept most stuff in this one call
        '''
        summary: the resulting summary as a list of sentences 
                 (NOTE: assumes that these sentences are most important)
        documentset: The set of documents sentences are extracted from (list of tokenized sentences)
                     Be sure to use DETOKENIZER.detokenize to make the docset a list of strings
                     (One string for each document)
        '''
        if isinstance(summary[0], list):
            summary = [DETOKENIZER.detokenize(s) for s in summary]

        summary_clone = deepcopy(summary)

        seen_clusters = {}
        _docset = [detokenize_list_of_tokens(doc) for doc in document_set]
        docset = list(map(self.resolver, tqdm(_docset, desc="Coreference resolution:")))

        for s_i, sentence in enumerate(summary):
            
            # recover the document
            reference_sentence = None
            for doc_i, doc in enumerate(docset):
                for j, s in enumerate(doc.sents):
                    if s.text in sentence:
                        reference_sentence = s
                        break
                if reference_sentence:
                    break
            
            # perform the algorithm
            seen_clusters[doc_i] = {}
            noun_phrases = [span for span in reference_sentence.noun_chunks]
            seen_nps = []
            for NP in noun_phrases:
                seen_nps.append(NP.text)
                for cluster_i in docset[doc_i].spans:
                    evaluate_tokens = lambda x, y: x.text == y.text and x.sent.text == y.sent.text
                    entity_in_list_of_spans = any([evaluate_tokens(NP, sp) for sp in doc.spans[cluster_i]])
                    if entity_in_list_of_spans:
                        
                        new_cluster = cluster_i not in seen_clusters[doc_i]

                        if new_cluster:
                            corefcluster = ReferenceClusters(doc, cluster_i)
                            replace_np = corefcluster.longest

                        else:
                            corefcluster = seen_clusters[doc_i][cluster_i]
                            replace_np = corefcluster.shortest
                            
                        if replace_np:
                            same_length = len(replace_np.text) == len(NP.text)
                            replacement_contain_pronouns = contains_a_pronoun(replace_np)
                            if same_length and replacement_contain_pronouns: # Don't replace NPs with the same length unless NP contains a pronoun and replacement doesn't
                                continue
                            # count previously seen NPs to replace the correct one in replace_nth function
                            np_count = seen_nps.count(NP.text)
                            summary_clone[s_i], replaced = replace_nth(summary_clone[s_i], NP.text, 
                                                            replace_np.text, np_count)
                            if replaced:
                                # Add to seen clusters only if completed replacement
                                seen_clusters[doc_i][cluster_i] = corefcluster
                                # # Print changes to the summary
                                # print(f"({np_count})", NP.text, "=>", replace_np.text)
                                # print(summary[s_i], "=>", summary_clone[s_i])

        return summary_clone
