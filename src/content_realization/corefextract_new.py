'''
This file contains a class that performs the coreference resolution
The implementation here is inspired by Nenkova (2007)
'''

from typing import *
import spacy
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.metrics.distance import jaccard_distance
import time
from copy import deepcopy
from utils import flatten_list
import re
import sys

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
    

PROPERNOUNTAGS={"PROPN", "NNP", "NNPS", "NE", "NNE", "NR"}
PRONOUNTAGS={"PRP", "PRON", "PRP$"}
DTTAGS={"DT", "WDT", "DET"}


class ReferenceClusters:

    def __init__(
            self, 
            doc: str, 
            cluster: str,
            reference_np: str,
            pronoun_tags = PRONOUNTAGS,
            determiner_tags = DTTAGS
        ) -> None:
        '''Class to hold coreference solutions and extract longest/shortest values'''
        self.doc = doc
        self.cluster = cluster
        self.pronoun_tags = pronoun_tags
        self.determiner_tags = determiner_tags
        self.reference_np = reference_np
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
            short_or_has_pronoun = len(span) == 1 or contains_a_pronoun(span, self.pronoun_tags)
            if not short_or_has_pronoun:
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

    def sort_by_length(self, filtered: bool = True, reverse: bool = False):
        arr = self.spans if filtered else self._spans
        tuple_arr = [(span, len(span.text)) for span in arr]
        sorted_arr = sorted(tuple_arr, reverse=reverse, key=lambda x: x[1])
        return [s[0] for s in sorted_arr]

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
        
    @property
    def _longest(self):
        lengths = self._lengths
        # if more than one value
        if len(lengths) > 0:
            max_length = max(lengths)
            argmax = lengths.index(max_length)
            longest = self._spans[argmax]
            return longest
        else:
            return None


def verify_element(nounp: "Span", subnp: "Span"):
    # list of cases to filter out changes
    gen_case_1 = nounp.text.endswith("'s") and subnp.text.endswith("'s")
    gen_case_2 = not nounp.text.endswith("'s") and not subnp.text.endswith("'s")
    same_case = gen_case_1 or gen_case_2
    if contains_a_pronoun(nounp) and contains_a_pronoun(subnp):
        same_case = False if nounp[0].morph != subnp[0].morph else same_case
    not_brackets = not nounp.text == "(" + subnp.text + ")"
    not_quotes = not nounp.text == "\"" + subnp.text + "\""
    same_proper_noun = True
    if contains_a_propernoun(nounp) and contains_a_propernoun(subnp):
        nounp_proper_nouns = set([tkn.text for tkn in nounp if tkn.tag_ in PRONOUNTAGS])
        subnp_proper_nouns = set([tkn.text for tkn in subnp if tkn.tag_ in PRONOUNTAGS])
        intersection = subnp_proper_nouns.intersection(nounp_proper_nouns)
        same_proper_noun = False if len(intersection) == 0 else True
    third_person_pn = True
    if contains_a_pronoun(subnp):
        third_person_pn = False if any(["Person=1" in str(tkn.morph) or "Person=2" in str(tkn.morph) for tkn in subnp]) else True
    return same_case and not_brackets and same_proper_noun and not_quotes and third_person_pn


def contains_a_pronoun(span, pronoun_tags: Set[str] = PRONOUNTAGS):
    return any([tkn.tag_ in pronoun_tags for tkn in span])


def all_pronouns(span, pronoun_tags: Set[str] = PRONOUNTAGS):
    return all([tkn.tag_ in pronoun_tags for tkn in span])


def contains_a_propernoun(span, propernp_tags: Set[str] = PROPERNOUNTAGS):
    return any([tkn.tag_ in propernp_tags for tkn in span])


def all_propernouns(span, propernp_tags: Set[str] = PROPERNOUNTAGS):
    return all([tkn.tag_ in propernp_tags for tkn in span])


def replace_via_indices(string: str, sub: str, start: int, end: int) -> str:
    '''Helper function to replace a string with another one at specified indices'''
    return "".join([string[:start], sub, string[end:]])


def replace_nth(string: str, sub: str, replacement: str, n: int):
    '''Replaces the string with the longest/shortest entity
       NOTE: It also corrects for replacements that use "I" in quotes. 
             When it is followed by a "said." string
    '''
    pattern_mid = re.compile(f'(?<=[\s\"\']){sub.text}(?=[\s\"\'\.,?!])')
    pattern_start = re.compile(f'^{sub.text}(?=[\s\"\'\.,?!])')
    repl_str = replacement.text
    # fix spelling
    if replacement[0].tag_ not in PROPERNOUNTAGS:
        repl_str = repl_str[0].lower() + repl_str[1:]
    if sub.text[0] in string.split()[0]:
        repl_str = repl_str[0].upper() + repl_str[1:]
    if re.match(pattern_start, string):
        split = re.split(pattern_start, string)
        start, end = f"{repl_str}".join(split[0:n]), f"{repl_str}".join(split[n+1:])
    else:
        split = re.split(pattern_mid, string)
        start, end = f"{repl_str}".join(split[0:n]), f"{repl_str}".join(split[n+1:])
    is_quote = start.count("\"") % 2 == 1 and end.count("\"") % 2 == 1
    replacing_I = sub.text == "I" and re.search(f".*said", string) is not None
    if is_quote and replacing_I:
        return string, False
    if is_quote and not replacing_I:
        repl_str = f"[{repl_str}]"
        start, end = f"{repl_str}".join(split[0:n]), f"{repl_str}".join(split[n+1:])
    return start + end, True


class CoferenceResolver:
    try:
        nlp = spacy.load("en_coreference_web_trf")
    except OSError:
        print("Cannot load \"en_coreference_web_trf\". Download it with: "
              "pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl")

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
    
    def __init__(self, document_set: List[List[List[str]]]) -> None:
        self.resolver = CoferenceResolver()
        
        if isinstance(document_set[0][0][0], list): # if the document_set contains paragraphs, remove them
            docset = []
            for doc in document_set:
                if isinstance(doc[0][0], list):
                    docset.append(flatten_list(doc))
                else:
                    docset.append(doc)
            document_set = docset
        
        document_set = " ".join([DETOKENIZER.detokenize(d) for d in document_set])

        start = time.time()
        self.docset = self.resolver(document_set)
        print(f"\tCoreference time: {time.time() - start}", file=sys.stderr)
        self.seen_clusters = {}

    def __call__(
            self, 
            sentence: Union[List[str], str]
        ) -> List[List[str]]:
        '''
        Args
            sentence: the resulting summary as a list of sentences 
                    (NOTE: assumes that these sentences are most important)
            documentset: The set of documents sentences are extracted from (list of tokenized sentences)
                        Be sure to use DETOKENIZER.detokenize to make the docset a list of strings
                        (One string for each document)
        '''
        # Sorry if the function is long... SpaCy's garbage collection is pretty aggresive towards 
        # out-of-scope variables, so I kept most stuff in this one call
        if isinstance(sentence, list):
            _sentence = DETOKENIZER.detokenize(sentence)
        
        # Iterate through a document set to recover the original indices of the sentences
        # If there are repeated sentences across documents, pick the first one.
        reference_sentence = None
        for j, s in enumerate(self.docset.sents):
            jaccard_dist_is_zero = jaccard_distance(set(word_tokenize(s.text)), set(sentence)) == 0.
            if s.text == _sentence or jaccard_dist_is_zero:
                reference_sentence = s
                break
        
        # perform the algorithm
        if reference_sentence is None:
            print(f"\tWarning (not in docset) -- "
                  f"Couldn't compelete coreference resolution for sentence: {_sentence}", 
                  file=sys.stderr)
            return sentence, False
        
        noun_phrases = [span for span in reference_sentence.noun_chunks]

        seen_nps = []
        switched = False
        for NP in noun_phrases:
            seen_nps.append(NP.text)
            for cluster_i in self.docset.spans:
                evaluate_tokens = lambda x, y: x.text == y.text and x.sent.text == y.sent.text
                entity_in_list_of_spans = any(
                    [evaluate_tokens(NP, sp) for sp in self.docset.spans[cluster_i]]
                )
                if entity_in_list_of_spans:
                    
                    replace_np = None

                    new_cluster = cluster_i not in self.seen_clusters

                    if new_cluster:
                        corefcluster = ReferenceClusters(self.docset, cluster_i, NP.text)
                        # first try longest premodified
                        longest_nps = corefcluster.sort_by_length(reverse=True, filtered=True)
                        for repl_np in longest_nps:
                            if contains_a_propernoun(NP) and not contains_a_propernoun(repl_np):
                                continue
                            elif len(repl_np.text) == len(NP.text):
                                continue
                            elif verify_element(NP, repl_np):
                                replace_np = repl_np
                        
                        # If no longer replacement found
                        if replace_np is not None: 
                            longest_nps = corefcluster.sort_by_length(reverse=True, filtered=False)
                            for repl_np in longest_nps:
                                if len(repl_np.text) == len(NP.text):
                                    continue
                                elif verify_element(NP, repl_np):
                                    replace_np = repl_np

                    else:
                        corefcluster = self.seen_clusters[cluster_i]
                        shortest_nps = corefcluster.sort_by_length(reverse=False, filtered=False)
                        for repl_np in shortest_nps:
                            if len(repl_np.text) >= len(NP.text):
                                break # no point in continuing search if existing is the shortest
                            elif verify_element(NP, repl_np):
                                replace_np = repl_np
                        
                    if replace_np:
                        same_length = len(replace_np.text) == len(NP.text)
                        is_a_pronoun = all_pronouns(replace_np)
                        if same_length or is_a_pronoun: # Don't replace NPs with pronouns
                            continue
                        # count previously seen NPs to replace the correct one in replace_nth function
                        np_count = seen_nps.count(NP.text) + 1
                        _sentence, replaced = deepcopy(replace_nth(_sentence, NP, 
                                                        replace_np, np_count))
                        if replaced:
                            switched = True
                            # Add to seen clusters only if completed replacement
                            self.seen_clusters[cluster_i] = corefcluster
                            # Print changes to the summary
                            seen = "unseen" if new_cluster else "seen"
                            # print(f"({seen})", NP.text, "=>", replace_np.text)
                            # print(DETOKENIZER.detokenize(sentence), "=>", _sentence)
        
        replacement = DETOKENIZER.detokenize(sentence) if switched else None

        return word_tokenize(_sentence), replacement
