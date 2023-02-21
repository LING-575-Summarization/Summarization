"""
From https://github.com/fabrahman/ReBART/blob/main/source/common.py

MIT License

Copyright (c) 2021 FaezeBr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
import random

from transformers import AutoModelWithLMHead, AutoTokenizer
from nltk.tokenize import word_tokenize


def init_model(model_name: str, device, do_lower_case: bool = False, args=None):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :param do_lower_case: whether the model is lower cased or not
    :return: the model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case, use_fast=False)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def get_json(json_path: str):
    with open(json_path) as json_file:
        return json.load(json_file)


def shuffle(input_json):
    result = []
    for docset in input_json:
        input_texts = docset["text"]
        for input_text in input_texts:
            input_text_result = []
            total_token = 3
            for count, value in enumerate(input_text):
                token_length = len(word_tokenize(value))
                if total_token + token_length + 1 < 800:
                    input_text_result.append((count, value))
                    total_token = total_token + token_length + 1
                else:
                    break
            random.shuffle(input_text_result)
            input_text_result = [list(t) for t in zip(*input_text_result)]
            result.append(input_text_result)
    return result


def combine_data(json_path: str, dataset_type: str):
    input_json = get_json(json_path)
    examples = []
    for docset in input_json[dataset_type]:
        input_texts = docset["text"]
        examples.append(
            (
                f"[shuffled] {' '.join([' '.join((f'<S{i}>', sent)) for i, sent in zip(list(range(len(input_texts))), input_texts)])} [orig]"
            )
        )
    return examples


def load_data(json_path: str):
    """
    Loads the dataset file:
    json_path: json file
    Returns a list of tuples (input, output)
    """
    input_json = get_json(json_path)
    all_lines = shuffle(input_json)
    examples = []
    for line in all_lines:
        if len(line) == 2:
            examples.append(
                (
                    f"[shuffled] {' '.join([' '.join((f'<S{i}>', sent)) for i, sent in zip(list(range(len(line[0]))), line[1])])} [orig]",
                    f"{' '.join([str(j) for j in line[0]])} <eos>",
                )
            )
    return examples


if __name__ == "__main__":
    test = load_data("/Users/junyinchen/Developer/Summarization/data/devtest.json")
    print(len(test))
