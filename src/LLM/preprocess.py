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
    for docset_id, docset in input_dict.items():
        print(docset)
        input_texts = docset["text"]
        for input_text in input_texts:
            print(input_text)
            for i in range(0, len(input_text)):
                current_sentence = input_text[i][0]
                print(current_sentence)
                gold = ""
                first_sentence = True
                for j in range(0, len(input_text)):
                    new_sentence = input_text[j][0].strip()
                    if j != i:
                        if first_sentence:
                            gold = new_sentence + "\n"
                            first_sentence = False
                        else:
                            gold = gold + "\n" + new_sentence
                print(gold)
                print(cal_metric(current_sentence, gold)["rouge1"].fmeasure)
            break
        break
    return input_dict


if __name__ == "__main__":
    result = get_json("../../data/training.json")
    result = find_best_sentence(result)
