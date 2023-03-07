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
        "--seed",
        type=int,
        default=42,
        help="Specify the random seed."
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default=get_root_dir() + "data/dataset.json",
        help="Path to JSON file containing raw dataset",
    )
    parser.add_argument(
        "--do_train",
        action='store_true',
        help="Train the model and store to default directory",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=get_root_dir() + "outputs/D5/LLM/pytorch_model.bin",
        help="Path to stored model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=get_root_dir() + "outputs/D4/LLM",
        help="Path to store predictions",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=get_root_dir() + "outputs/D4/",
        help="Path to store results"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=12,
        help="Number of training epoch",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="junyinc/LING-575-WI-SUM",
        help="Which checkpoint to use."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="How many examples per batch"
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=120,
        help="How many examples per batch"
    )
    parser.add_argument(
        "--raw_json_dir",
        type=str,
        default="data/",
        help="Path to store predictions",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Which Model tage should be used",
    )
    args, unknown = parser.parse_known_args()
    return args
