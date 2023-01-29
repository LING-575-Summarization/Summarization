import random

import json
from rouge_score import rouge_scorer


def get_json(json_path: str):
    with open(json_path) as json_file:
        return json.load(json_file)


def cal_metric(prediction, target):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(target, prediction)


def build_dataset(input_dict, dataset_type):
    output = []
    for docset_id, docset in input_dict.items():
        output_dict = dict()
        output_dict["id"] = docset_id
        output_dict["title"] = docset["title"]
        input_texts = docset["text"]
        output_dict["text"] = []
        for input_text in input_texts:
            if dataset_type == "training":
                output_dict["text"].append(mask_sentences(input_text))
            else:
                output_dict["text"].append(" \n ".join([]))
        output_dict["summary"] = docset["summary"][random.randint(0, len(docset["summary"]) - 1)]
        output.append(output_dict)
    return output


def mask_sentences(input_text):
    scores = get_sentence_score(input_text)
    indexes = generate_index_list(len(input_text))
    top_30_percent = len(input_text) * 3 // 10
    compare = [x for _, x in sorted(zip(scores, indexes))][:top_30_percent]
    output = ""
    for i in range(0, len(input_text)):
        if i in compare:
            output = output + " \n " + "[MASK]"
        else:
            output = output + " \n " + input_text[i]
    return output.strip()


def generate_index_list(size: int):
    output = []
    index = 0
    while index < size:
        output.append(index)
        index += 1
    return output


def get_sentence_score(input_text):
    score = []
    for i in range(0, len(input_text)):
        current_sentence = input_text[i]
        gold = ""
        for j in range(0, len(input_text)):
            new_sentence = input_text[j].strip()
            if j != i:
                gold = gold + "\n" + new_sentence
        score.append(cal_metric(current_sentence, gold.strip())["rouge1"].fmeasure)
    return score


def preprocess(input_path, dataset_type):
    input_json = get_json(input_path)
    return build_dataset(input_json, dataset_type)


if __name__ == "__main__":
    with open("../../data/training_ds.json", "w") as final:
        json.dump(preprocess("../../data/training.json", "training"), final)
    with open("../../data/validation_ds.json", "w") as final:
        json.dump(preprocess("../../data/evaltest.json", "validation"), final)
    with open("../../data/test_ds.json", "w") as final:
        json.dump(preprocess("../../data/devtest.json", "test"), final)
