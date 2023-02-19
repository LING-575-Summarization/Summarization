import argparse
import random

import json
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

import util


def get_json(json_path: str):
    with open(json_path) as json_file:
        return json.load(json_file)


def cal_metric(prediction, target):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    return scorer.score(target, prediction)


def build_dataset(input_dict, dataset_type, args):
    output = []
    for docset in input_dict:
        docset_id = docset["id"]
        input_texts = docset["text"] if not args.reordered else docset["reordered"]
        total_length = 0
        over_limit = False
        whole_text = "" if not args.do_reorder else []
        for input_text in input_texts:
            if over_limit:
                break
            new_doc, total_length, over_limit = mask_sentences(input_text, dataset_type, total_length,
                                                               over_limit, docset["summary"], args)
            whole_text = whole_text + " " + new_doc if not args.do_reorder else whole_text.append(new_doc)

        if dataset_type == "training" and not args.do_reorder:
            for count, summary in enumerate(docset["summary"]):
                output_dict = dict()
                output_dict["id"] = docset_id + "_" + count
                output_dict["text"] = whole_text
                output_dict["summary"] = summary
                output.append(output_dict)
        else:
            output_dict = dict()
            output_dict["id"] = docset_id
            output_dict["text"] = whole_text
            output_dict["summary"] = docset["summary"][random.randint(0, len(docset["summary"]) - 1)]
            output.append(output_dict)
    return output


def mask_sentences(input_text, dataset_type, total_length, over_limit, gold_list, args):
    scores = get_sentence_score(input_text, dataset_type, gold_list)
    indexes = generate_index_list(len(input_text))
    top_30 = [x for _, x in sorted(zip(scores, indexes))][:len(input_text) * 2 // 10]
    low_50 = [x for _, x in sorted(zip(scores, indexes))][:len(input_text) * 5 // 10]
    output = "" if not args.do_reorder else []
    previous_mask = False
    for i in range(0, len(input_text)):
        if over_limit:
            break
        if i in top_30 and dataset_type == "training" and args.do_mask:
            if previous_mask:
                continue
            else:
                output = output + " " + "[MASK]"
                total_length += 1
                previous_mask = True
        elif i in low_50:
            continue
        else:
            token_length = len(word_tokenize(input_text[i]))
            if total_length + token_length > 1024:
                over_limit = True
                break
            if not args.do_reorder:
                output = output + " " + input_text[i]
            else:
                output.append(input_text[i])
            total_length += token_length
            previous_mask = False
    if not args.do_reorder:
        output = output.strip()
    return output, total_length, over_limit


def generate_index_list(size: int):
    output = []
    index = 0
    while index < size:
        output.append(index)
        index += 1
    return output


def get_sentence_score(input_text, dataset_type, gold_list):
    score = []
    for i in range(0, len(input_text)):
        current_sentence = input_text[i]
        if dataset_type == "training":
            score_list = []
            for gold_summary in gold_list:
                score_list.append(cal_metric(current_sentence, gold_summary.strip())["rouge1"].fmeasure)
            score.append(sum(score_list) / len(score_list))
        else:
            gold = ""
            for j in range(0, len(input_text)):
                new_sentence = input_text[j].strip()
                if j != i:
                    gold = gold + "\n" + new_sentence
            score.append(cal_metric(current_sentence, gold.strip())["rouge1"].fmeasure)
    return score


def preprocess(input_path, dataset_type, args):
    input_json = get_json(input_path)
    return build_dataset(input_json, dataset_type, args)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_json_dir",
        default="data/",
        type=str,
        help="Input directory for data json ran with preprocess.",
    )

    parser.add_argument(
        "--rouge", default="rouge1", type=str, help="Which rouge you want to use"
    )
    parser.add_argument(
        "--do_reorder",
        action="store_true",
        help="Set this flag if you want to produce jsons for reordering.",
    )
    parser.add_argument(
        "--reordered",
        action="store_true",
        help="Set this flag if you want to produce jsons for reordering.",
    )
    parser.add_argument(
        "--do_mask",
        action="store_true",
        help="Set this flag if you want to do the masking.",
    )
    args, unknown = parser.parse_known_args()

    dir_prefix = util.get_root_dir() + args.raw_json_dir
    dataset = dict()
    dataset["train"] = preprocess(dir_prefix + "training.json", "training", args)
    dataset["validation"] = preprocess(dir_prefix + "evaltest.json", "validation", args)
    dataset["test"] = preprocess(dir_prefix + "devtest.json", "test", args)
    with open(dir_prefix + "dataset.json", "w") as final_output:
        json.dump(dataset, final_output)


if __name__ == "__main__":
    main()
