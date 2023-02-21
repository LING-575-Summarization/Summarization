"""
    # **WARNING**: Does not quite work, detokenizing/tokenizing can create
        KeyNotFound Error for sentence_to_embedding Dictionary
    # TODO: Fix the above bug
    This will read in the summaries in a directory and output ordered summaries
"""


import sys, os
from clustering import create_clusters, SentenceIndex
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def detokenize_summary(summary):
    detokenizer = TreebankWordDetokenizer()
    summary_str = ""
    for sent in summary:
        summary_str += detokenizer.detokenize(sent) + "\n"
    return summary_str


def order_summaries(summary_dir, json_file, output_dir):
    """
        Expects a summary, where one sentence is per line
    """
    fractional_order = SentenceIndex(json_file)
    for dirpath, dir, files in os.walk(summary_dir):
        for file in files[2:3]:
            file_path = os.path.join(dirpath, file)
            with open(file_path) as f:
                summary = f.readlines()
            tokenized_sum = []
            for sent in summary:
                tokenized_sum.append(word_tokenize(sent.strip()))

            print("tokenized summary")
            print(tokenized_sum)
            file_parts = file.split(".")
            id_part1 = file_parts[0][:-2]
            id_part2 = file_parts[-2]
            docset_id = id_part1 + id_part2 + "-A"
            summary = create_clusters(docset_id, tokenized_sum, fractional_order)

            print("ordered summary")
            print(summary)
            print("####################\n")
            summary = detokenize_summary(summary)
            output_file = os.path.join(output_dir, file)
            with open(output_file, "w") as output:
                output.write(summary)


if __name__ == '__main__':
    summary_dir = sys.argv[1]
    output_dir = sys.argv[2]
    json_file = sys.argv[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cur_directory = "."
    sum_files = os.path.join(cur_directory, summary_dir)

    order_summaries(summary_dir, json_file, output_dir)
