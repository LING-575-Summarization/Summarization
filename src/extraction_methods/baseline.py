import sys
import os
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from clustering import SentenceIndex, create_clusters
from content_realization import replace_referents
from utils import flatten_list, docset_loader


def detokenize_summary(summary):
    detokenizer = MosesDetokenizer()
    summary_str = ""
    for sent in summary:
        summary_str += detokenizer.detokenize(sent) + "\n"
    return summary_str


def make_summary(doc, max_summary_length):
    total_length = 0
    text = doc[3]
    summary = []
    for paragraph in text:
        for sentence in paragraph:
            sent_len = len(sentence)
            # get the first short enough sentence into the summary
            if summary == []:
                if (total_length + sent_len) <= max_summary_length:
                    summary.append(sentence)
                    total_length += sent_len
            # fill out as much as you can after
            elif (total_length + sent_len) <= max_summary_length:
                summary.append(sentence)
                total_length += sent_len
            else:
                return summary
    # return summary


def read_json(json_path):
    with open(json_path, "r") as final:
        docset_rep = json.load(final)
    return docset_rep

if __name__ == '__main__':
    # from shell script
    json_path = sys.argv[1]
    max_summary_length = int(sys.argv[2])
    output_dir = sys.argv[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    docset_rep = read_json(json_path)
    fractional_order = SentenceIndex(json_path)

    for docset_id, docset in docset_rep.items():
        for doc_id, doc in docset.items():
            # list of sentences, where each sentence is tokenized words
            summary = make_summary(doc, max_summary_length)

            # information ordering
            summary = create_clusters(docset_id, summary, fractional_order, json_path)

            # content realization
            docset_2, indices = docset_loader(json_path, docset_id)
            summary = replace_referents(summary, docset_2)

            # detokenize summary
            summary = detokenize_summary(summary)
            # output summaries to correct file
            docset_id = docset_id.split("-")[0]
            id_part1 = docset_id[:-1]
            id_part2 = docset_id[-1]
            output_summary_file = output_dir + "/" + id_part1 + "-A.M.100." + id_part2 + ".4"
            with open(output_summary_file, 'w') as output:
                output.write(summary)
            break




