"""
From https://github.com/fabrahman/ReBART/blob/main/source/generate.py

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

Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
"""
import argparse
import json
import logging
import re

import torch
import tqdm


from common import init_model, combine_data


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Generate outputs
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--in_file",
        default=None,
        type=str,
        required=True,
        help="The input json file",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        required=True,
        help="out jsonl file",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="LM checkpoint for initialization.",
    )

    # Optional
    parser.add_argument(
        "--max_length", default=1024, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=0, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--dataset_type",
        default="train",
        type=str,
        help="Which part of the dataset to load.",
    )

    args = parser.parse_args()
    logger.debug(args)

    if (
            (args.k == args.p == args.beams == 0)
            or (args.k != 0 and args.p != 0)
            or (args.beams != 0 and args.p != 0)
            or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(args.model_name_or_path, device)

    examples = combine_data(args.in_file, args.dataset_type)

    logger.info(examples[:5])

    special_tokens = ["[shuffled]", "[orig]", "<eos>"]
    extra_specials = [f"<S{i}>" for i in range(args.max_length)]
    special_tokens += extra_specials

    with open(args.out_file, "w") as f_out:
        for input_lines in tqdm.tqdm(examples):
            try:
                preds = generate_conditional(
                    tokenizer,
                    model,
                    args,
                    input_lines,
                    device,
                )

                # Remove any word that has "]" or "[" in it
                preds = [re.sub(r"(\w*\])", "", pred) for pred in preds]
                preds = [re.sub(r"(\[\w*)", "", pred) for pred in preds]
                preds = [re.sub(" +", " ", pred).strip() for pred in preds]

            except Exception as exp:
                logger.info(exp)
                preds = []

            f_out.write(
                json.dumps({"input": input_lines, "predictions": preds})
                + "\n"
            )


def generate_conditional(tokenizer, model, args, input, device):
    """
    Generate a sequence with models like Bart and T5
    """
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    decoder_start_token_id = input_ids[-1]
    input_ids = torch.tensor([input_ids]).to(device)
    max_length = args.max_length

    outputs = model.generate(
        input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        min_length=5,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=decoder_start_token_id,
        num_return_sequences=1  # max(1, args.beams)
    )

    preds = [tokenizer.decode(
        output, skip_special_tokens=False, clean_up_tokenization_spaces=False) for output in outputs]

    return preds


if __name__ == "__main__":
    main()
