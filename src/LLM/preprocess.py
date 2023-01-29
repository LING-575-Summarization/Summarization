import random

import json
from nltk.tokenize import word_tokenize
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
        # output_dict["title"] = docset["title"]
        input_texts = docset["text"]
        output_dict["text"] = ""
        total_length = 0
        over_limit = False
        for input_text in input_texts:
            if over_limit:
                break
            new_doc, total_length, over_limit = mask_sentences(input_text, dataset_type, total_length, over_limit)
            output_dict["text"] = output_dict["text"] + " <|endoftext|> " + new_doc
        print(total_length)
        print(output_dict["text"])
        output_dict["summary"] = docset["summary"][random.randint(0, len(docset["summary"]) - 1)]
        output.append(output_dict)
    return output


def mask_sentences(input_text, dataset_type, total_length, over_limit):
    scores = get_sentence_score(input_text)
    indexes = generate_index_list(len(input_text))
    top_30 = [x for _, x in sorted(zip(scores, indexes))][:len(input_text) * 2 // 10]
    low_30 = [x for _, x in sorted(zip(scores, indexes))][:len(input_text) * 5 // 10]
    output = ""
    previous_mask = False
    for i in range(0, len(input_text)):
        if over_limit:
            break
        if i in top_30 and dataset_type == "training":
            if previous_mask:
                continue
            else:
                output = output + " \n " + "[MASK]"
                total_length += 1
                previous_mask = True
        elif i in low_30:
            continue
        else:
            token_length = len(word_tokenize(input_text[i]))
            if total_length + token_length > 1024:
                over_limit = True
                break
            output = output + " \n " + input_text[i]
            total_length += token_length
            previous_mask = False
    return output.strip(), total_length, over_limit


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
    dataset = dict()
    dataset["train"] = preprocess("../../data/training.json", "training")
    dataset["validation"] = preprocess("../../data/evaltest.json", "validation")
    dataset["test"] = preprocess("../../data/devtest.json", "test")
    with open("../../data/dataset.json", "w") as final:
        json.dump(dataset, final)

