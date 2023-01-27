import argparse
from os.path import abspath, dirname


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_tokenize",
        action='store_true',
        help="Do not do word level tokenization, only return full sentence",
    )
    parser.add_argument(
        "--gold_directory",
        type=str,
        default="/dropbox/22-23/575x/Data/models/devtest",
        help="Path to the directory containing gold summary",
    )
    args, unknown = parser.parse_known_args()
    return args
