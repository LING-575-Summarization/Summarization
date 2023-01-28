import evaluate
import numpy as np
import json
from rouge_score import rouge_scorer


# rouge_metric = evaluate.load("rouge")


def get_json(json_path: str):
    with open(json_path) as json_file:
        return json.load(json_file)


def cal_metric(prediction, target):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(target, prediction)


def find_best_sentence(input_dict):
    print("Type:", type(input_dict))
    for docset_id, docset in input_dict.items():
        print(docset)
        gold = docset["summary"][0]
        input_texts = docset["text"]
        for input_text in input_texts:
            print(input_text)
            i = 0
            for sentence in input_text:
                print(sentence[0])
                print(cal_metric(sentence[0], gold))
                i += 1
            print(i)
            break
        break
    return input_dict


if __name__ == "__main__":
    result = get_json()
    result = find_best_sentence(result)
