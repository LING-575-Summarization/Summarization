import argparse
from os.path import abspath, dirname


def get_root_dir():
    result = dirname(abspath(__file__))
    src = "/src/LLM/Summarize"
    print(result)
    if result.endswith(src):
        result = result[:-len(src) + 1]
    return result


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        required=True,
        help="Out directory for checkpoints.",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_data_file",
        type=str,
        required=True,
        help="The input CSV validation file."
    )
    parser.add_argument(
        "--eval_during_train",
        action="store_true",
        help="Evaluate at each train logging step.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Steps before backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-6,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=-1,
        help="Log every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--max_input_length",
        default=1024,
        type=int,
        help="Maximum input event length in words.",
    )
    parser.add_argument(
        "--max_output_length",
        default=120,
        type=int,
        help="Maximum output event length in words.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: total number of training steps to perform.",
    )
    parser.add_argument(
        "--model_name",
        default="facebook/bart-base",
        type=str,
        help="LM checkpoint for initialization.",
    )
    parser.add_argument(
        "--model_type",
        default="",
        type=str,
        help="which family of LM, e.g. gpt, gpt-xl, ....",
    )
    parser.add_argument(
        "--epoch",
        default=2,
        type=int,
        help="Number of training epochs to perform.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached data."
    )
    parser.add_argument(
        "--overwrite_out_dir",
        action="store_true",
        help="Overwrite the output directory.",
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from the last checkpoint.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=False,
        help="The input CSV train file."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    args, unknown = parser.parse_known_args()
    return args
