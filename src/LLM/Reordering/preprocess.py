import json
import random


def shuffle(json_path):
    input_json = get_json(json_path)
    result = []
    for docset_id, docset in input_json.items():
        input_texts = docset["text"]
        for input_text in input_texts:
            input_text_result = []
            for count, value in enumerate(input_text):
                input_text_result.append((count, value))
            random.shuffle(input_text_result)
            input_text_result = [list(t) for t in zip(*input_text_result)]
            result.append(input_text_result)
    return result


def load_data(all_lines):
    """
    Loads the dataset file:
    in_file: json file
    Returns a list of tuples (input, output)
    """
    examples = [
        (
            f"[shuffled] {' '.join([' '.join((f'<S{i}>', sent)) for i, sent in zip(list(range(len(line[0]))), line[1])])} [orig]",
            f"{' '.join(line[0])} <eos>",
        )
        for line in all_lines
    ]
    return examples
