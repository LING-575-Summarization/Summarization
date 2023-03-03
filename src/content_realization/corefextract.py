'''
This file contains a class that performs the coreference resolution
The implementation here is inspired by Nenkova (2007)
'''

from typing import *
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils import detokenize_list_of_tokens
from collections import namedtuple
from copy import deepcopy
import re


# NP = namedtuple("NP", "text sentence_i sent start end")

DETOKENIZER = TreebankWordDetokenizer()

class ModifiedNP:

    def __init__(self, doc, cluster, pronoun_tags = {"PRP", "PRON", "PRP$", "DT"}) -> None:
       self.doc = doc
       self.cluster = cluster
       self.pronoun_tags = pronoun_tags

    def __iter__(self):
        return iter(self.doc.spans[self.cluster])

    def __post_init__(self):
        '''Remove any postpositions'''
        for i, span in enumerate(self.doc.spans[self.cluster]):
            tags = [tkn.tag_ for tkn in span]
            if "," in tags or "$," in tags:
                self.doc.spans[self.cluster].pop(i)

    def proper_spans(self):
        # remove unwanted NPs like pronouns
        filtered = []
        for span in self.doc.spans[self.cluster]:
            contains_a_pronoun = len(span) == 1 or len(set([tkn.tag_ for tkn in span]) & self.pronoun_tags) > 0
            if not contains_a_pronoun:
                filtered.append(span)
        return filtered

    @property
    def lengths(self):
        return [sp.text for sp in self.doc.spans[self.cluster]]

    @property
    def shortest(self):
        lengths = [sp.text for sp in self.proper_spans()]
        print("Min length:", self.lengths)
        if len(lengths) > 1:
            min_length = min(lengths)
            argmin = lengths.index(min_length)
            return self.doc.spans[self.cluster][argmin]
        else:
            return None

    def shortest_nonproper(self):
        min_length = min(self.lengths)
        argmin = self.lengths.index(min_length)
        return self.doc.spans[self.cluster][argmin]

    @property
    def longest(self):
        if len(self.lengths) > 1:
            max_length = max(self.lengths)
            argmax = self.lengths.index(max_length)
            return self.doc.spans[self.cluster][argmax]
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


def replace_via_indices(string: str, sub: str, start: int, end: int) -> str:
    '''Helper function to replace a string with another one at specified indices'''
    return "".join([string[:start], sub, string[end:]])


def replace_nth(string: str, sub: str, replacement: str, n: int, quote: Optional[bool] = False):
    pattern = re.compile(f'(?<=[\s\"\']){sub}(?=([\s\"\,\.\?\!]|\'\s))')
    split = re.split(pattern, string)
    replacement = replacement if not quote else f"[{replacement}]"
    return f"{sub}".join(split[0:n]) + replacement + f"{sub}".join(split[n:])


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


# NEW APPROACH: NEED TO FIND A WAY TO CONTINUE POINTING TO THE RIGHT REFERENT BEFORE IT'S
#               GROUPED INTO A SUMMARY. MIGHT NEED TO ASSIGN EACH WORD A POINTER of some sort.
#               When it's processed by LexRank, you can just go back and find out where it was
#               clustered into.

# ANOTHER APPROACH IS TO FIRST PARSE THE FILES FOR NPs. 
# 1. GO THROUGH EACH SENTENCE:
# 2. PERFORM A ENTITY PARSE ON THE PARAGRAPH IT IS FOUND IN. FIND THE LONGEST PREMODIFIED NP.
#       2A. AND REPLACE EVERY ENTITY WITH THE LONG ONE. 
#       2B. SAVE A DICTIONARY OF VALID NPs FOR EACH NP INVESTIGATED.
# 3. PERFORM A PARSE ON THE SUMMARY TO FIND INTERNAL COREFERENCES. IF A COREFERENCE IS PREVIOUSLY
#    MENTIONED, REPLACE IT WITH THE SHORTEST ONE FROM THE COMPILED DICIONTARY.

class ContentRealizer:
    resolver = CoferenceResolver()

    def __call__(self, summary: List[List[str]], 
                 document_set: List[List[str]]) -> None:
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

        corresponding_nps = {}
        _docset = [detokenize_list_of_tokens(doc) for doc in document_set]
        docset = list(map(self.resolver, _docset))

        for s_i, sentence in enumerate(summary):
            corresponding_nps[s_i] = {}
            # recover the document
            reference_sentence = None
            for i, doc in enumerate(docset):
                for j, s in enumerate(doc.sents):
                    if s.text in sentence:
                        reference_sentence = s
                        break
                if reference_sentence:
                    break
            
            noun_phrases = [span for span in reference_sentence.noun_chunks]
            seen_nps = []
            for NP in noun_phrases:
                seen_nps.append(NP.text)
                for cluster_i in docset[i].spans:
                    evaluate_tokens = lambda x, y: x.text == y.text and x.sent.text == y.sent.text
                    entity_in_list_of_spans = any([evaluate_tokens(NP, sp) for sp in doc.spans[cluster_i]])
                    if entity_in_list_of_spans:
                        corresponding_nps[s_i][NP.text] = [w for w in doc.spans[cluster_i]]
                        lengths = [len(s.text) for s in corresponding_nps[s_i][NP.text]]
                        max_np_length = max(lengths)
                        longest_np_i = lengths.index(max_np_length)
                        longest_np = corresponding_nps[s_i][NP.text][longest_np_i]
                        quote = True if all([t.is_quote for t in longest_np]) else False
                        np_count = seen_nps.count(NP.text)
                        summary_clone[s_i] = replace_nth(summary_clone[s_i], NP.text, longest_np.text, np_count, quote)
                        print(f"({np_count})", NP.text, "=>", longest_np)
                        print(summary[s_i], "=>", summary_clone[s_i])

        # TODO: More filtering for greater premodified sequence. More debugging.

        print(summary_clone)
        return summary_clone



class _ContentRealizer:
    resolver = CoferenceResolver()

    def __call__(self, summary: List[List[str]], 
                 document_set: List[List[str]]) -> None:
        '''
        summary: the resulting summary as a list of sentences 
                 (NOTE: assumes that these sentences are most important)
        documentset: The set of documents sentences are extracted from (list of tokenized sentences)
                     Be sure to use DETOKENIZER.detokenize to make the docset a list of strings
                     (One string for each document)
        '''
        if isinstance(summary[0], list):
            summary = [DETOKENIZER.detokenize(s) for s in summary]

        NPs, resolved_nps, seen_clusters = [], {}, {}
        new_summary = deepcopy(summary)
        for s_i, sentence in enumerate(summary):
            parsed_s = self.resolver.parser(sentence)
            noun_phrases = [NP(span.text, s_i, sentence, span.start, span.end) for span in parsed_s.noun_chunks]
            NPs.extend(noun_phrases)

        raw_document = detokenize_list_of_tokens(document_set)
        doc = self.resolver(raw_document)

        for nounp in NPs:
            print("LOOKING AT NP", nounp.text)
            desired_cluster = None

            for cluster_i in doc.spans:
                entity_in_list_of_spans = any([nounp.text == sp.text for sp in doc.spans[cluster_i]])
                if entity_in_list_of_spans:
                    desired_cluster = cluster_i
                    print("FOUND NP IN CLUSTER", doc.spans[cluster_i])

            if desired_cluster is None:
                continue
            elif desired_cluster in seen_clusters:
                shortest = resolved_nps[desired_cluster].shortest
                print("Tags", [w.tag_ for w in shortest])
                if shortest:
                    new_sent = new_summary[nounp.sentence_i].replace(nounp.text, shortest.text)
                    new_summary[nounp.sentence_i] = new_sent
                    print(nounp.text, "=>", shortest, f"({new_summary[nounp.sentence_i]})")
            else:
                coreferences = ModifiedNP(doc, desired_cluster)
                longest = coreferences.longest
                if longest:
                    print("Tags", [w.tag_ for w in longest])
                    longest = longest.text
                    # add brackets when in quotes
                    if nounp.is_quote:
                        longest = f"[{longest}]"
                    new_sent = new_summary[nounp.sentence_i].replace(nounp.text, longest)
                    print(nounp.text, "=>", longest, f"({new_summary[nounp.sentence_i]})")
                    new_summary[nounp.sentence_i] = new_sent
                    resolved_nps[cluster_i] = coreferences
        
        return new_summary

