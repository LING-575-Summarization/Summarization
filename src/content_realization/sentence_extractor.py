'''
This file contains a function to extact sentences from a ranked list of sentences.
The tokenize wrapper is used to detokenize sentences before putting them in the summary.
The function is optionally combined with the coreference extactor to generate summaries.
'''

from utils import detokenizer_wrapper
from nltk.metrics.distance import jaccard_distance
from .corefextract import ContentRealizer
from typing import *
import re


@detokenizer_wrapper
def extract_summary(
        ranked_list: List[List[str]],
        reference_documents: Optional[List[List[str]]] = None,
        max_tokens: Optional[int] = 100,
        topk_sentences: Optional[int] = 25,
        min_jaccard_dist: Optional[float] = 0.,
        coreference_resolution: Optional[bool] = True, 
        detokenize: Optional[Union[Callable, bool]] = False
    ) -> Union[str, List[List[str]]]:
    '''
    Obtain a "summary" of the document by running the method `solve_lexrank`
    and then selecting sentences until it reaches the max number of words
    Arguments:
        - ranked_list: a ranked list of extracted sentences formatted as a list of tokens
        - reference_documents: the document set being analyzed
        - max_words: max tokens in the summary
        - topk_sentences: consider only sentences in the topk for the summary
        - min_jaccard_dist: the minimum Jaccard distance (1 is most dissimilar) required in a new
                            sentence to include in the summary
        - coreference_resolution: whether or not to perform coreference resolution on the sentence
        - detokenize: whether to combine the tokens into a typical English sentence
            or leave as a list of whitespace delimited tokens. The decorator 
            wrap_detokenizer transforms the tokenize bool into a function behind the scenes
    '''
    i = 0
    words = 0
    summary_ids = []
    number_of_sentences = len(ranked_list)
    if coreference_resolution:
        assert reference_documents is not None, "coreference_resolution requires reference documents"
        cr = ContentRealizer(reference_documents)
    else:
        cr = lambda x: (x, None)
    while words < max_tokens and i < min(topk_sentences, number_of_sentences):
        # filter unfinished quotes
        if re.match(r'^\"[^\"]+$|^[^\"]+\"$', " ".join(ranked_list[i])):
            i += 1
            continue
        current_sentence, _ = cr(ranked_list[i])
        if min_jaccard_dist is not None:
            too_similar = False
            for previous_sent_id in summary_ids:
                prev_sent = ranked_list[previous_sent_id]
                jaccard_d = jaccard_distance(set(current_sentence), set(prev_sent))
                if jaccard_d <= min_jaccard_dist:
                    too_similar = True
                    break
        if min_jaccard_dist is not None and too_similar:
            i += 1
            continue
        if len(current_sentence) + words > max_tokens:
            i += 1
            continue
        else:
            summary_ids.append(i)
            words += len(current_sentence)
            i += 1
    summary = [detokenize(ranked_list[sum_id]) for sum_id in summary_ids]
    assert words - len(current_sentence) < max_tokens, \
        f"words: {words - len(current_sentence)} | sentence: {current_sentence}"
    if isinstance(summary[-1], str):
        summary = list(map(lambda x: x + "\n", summary))
    return detokenize(summary)