import argparse
from os.path import abspath, dirname


def get_root_dir():
    result = dirname(abspath(__file__))
    src = "/src"
    if result.endswith(src):
        result = result[:-len(src) + 1]
    return result


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_tokenize",
        action='store_false',
        help="Do not do word level tokenization, only return full sentence",
    )
    args, unknown = parser.parse_known_args()
    return args
