'''
This file contains a class that performs the coreference resolution
The implementation here is inspired by Nenkova (2007)
'''

from typing import *
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from preprocess.tokenizer import SentenceTokenizer
from utils import detokenize_list_of_tokens
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass


NP = namedtuple("NP", "text sentence_i start_index end_index")

DETOKENIZER = TreebankWordDetokenizer()
TOKENIZER = SentenceTokenizer()

@dataclass
class ModifiedNP:
    spans: List["Span"]
    pronoun_tags: Set[str] = {"PRP", "PRON", "PRP$", "DT"}

    def __post_init__(self):
        '''Remove any postpositions'''
        for i, span in enumerate(self.spans):
            tags = [tkn.tag_ for tkn in span]
            if "," in tags or "$," in tags:
                index = tags.index(",") if tags.index(",") else tags.index("$,")
                self.spans[i] = span[:index+1]

    def proper_spans(self):
        # remove unwanted NPs like pronouns
        filtered = []
        for span in self.spans:
            just_a_pronoun = len(span) == 1 and len(set([tkn.tag_ for tkn in span]) & self.pronoun_tags) > 0
            if not just_a_pronoun:
                filtered.append(span)
        return filtered

    @property
    def lengths(self):
        return [sp.text for sp in self.spans]

    def get_shortest(self, noun_phrase):
        try_pn_first = self.shortest_nonproper()
        if noun_phrase.sentence_i == try_pn_first.sent:
            return try_pn_first
        else:
            return self.shortest_proper()

    def shortest_nonproper(self):
        min_length = min(self.lengths)
        argmin = self.lengths.index(min_length)
        return self.spans[argmin]
    
    def shortest_proper(self):
        lengths = [sp.text for sp in self.proper_spans()]
        min_length = min(lengths)
        argmin = lengths.index(min_length)
        return self.spans[argmin]

    @property
    def longest(self):
        max_length = max(self.lengths)
        argmax = self.lengths.index(max_length)
        return self.spans[argmax]



def recover_document(sentence: List[str], document_set: List[List[str]]) -> Tuple[int, int]:
    '''
    Iterate through a document set to recover the original indices of the sentences
    The document set should be a list of a list of sentences. Can use docset_loader to reload
        sentences if you'd like
    '''
    for doc in document_set:
        for j, s in enumerate(doc):
            if sentence == s:
                return (doc, j)
    return None


def replace_via_indices(string: str, sub: str, start: int, end: int) -> str:
    '''Helper function to replace a string with another one at specified indices'''
    return "".join([string[:start], sub, string[end:]])


class CoferenceResolver:
    nlp = spacy.load("en_coreference_web_trf")
    parser = spacy.load("en_core_web_md")

    def __call__(self, doc: str, entity: Optional[str] = None) -> Any:
        '''
        If a string is provided, return the entities that match this string. Otherwise, return
        all the entity clusters in the documemt
        '''
        spacydoc = self.nlp(self.parser(doc))
        if entity is None:
            return spacydoc.spans
        else:
            for cluster_i, coref in spacydoc.spans.items():
                if entity in set([sp.text for sp in coref]):
                    return coref

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

    def __call__(self, summary: List[str], document_set: List[List[List[str]]]) -> None:
        '''
        summary: the resulting summary as a list of sentences 
                 (NOTE: assumes that these sentences are most important)
        documentset: The set of documents sentences are extracted from (list of tokenized sentences)
                     Be sure to use DETOKENIZER.detokenize to make the docset a list of strings
                     (One string for each document)
        '''
        NPs, resolved_nps = [], {}
        new_summary = deepcopy(summary)
        for s_i, sentence in enumerate(summary):
            parsed_s = self.parser(sentence)
            noun_phrases = [NP(span.text, s_i, span.start, span.end) for span in parsed_s.noun_chunks]
            NPs.extend(noun_phrases)
        for NP in NPs:
            if NP.text in resolved_nps:
                shortest = resolved_nps[NP.text].get_shortest(NP.text)
                new_sent = replace_via_indices(new_summary[NP.sentence_i], 
                                               shortest, NP.start, NP.end)
                new_summary[NP.sentence_i] = new_sent
            else:
                doc, s_j = recover_document(NP.sentence_i, document_set)
                raw_document = detokenize_list_of_tokens(doc)
                coreferences = ModifiedNP(self.resolver(raw_document, NP.text))
                longest = coreferences.longest
                new_sent = replace_via_indices(new_summary[NP.sentence_i], 
                                               longest, NP.start, NP.end)
                for ref in coreferences:
                    resolved_nps[ref.text] = coreferences
        return new_summary

