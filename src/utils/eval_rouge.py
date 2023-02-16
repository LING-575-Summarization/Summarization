import rouge
from typing import List, Optional
from pathlib import Path
from collections import OrderedDict
import os, sys
from itertools import product as cartesian_product
import re
import pandas as pd
import argparse
from typing import *


user_root = os.path.expanduser("~")
EVALFILES = os.path.join(user_root, "/dropbox/22-23/575x/Data/models/devtest/")
SUMFILES = os.path.join(user_root, "575-Summarization", "outputs", "D4")
ROUGE_ARGS = dict(
    metrics=['rouge-n'],
    max_n=2,
    limit_length=True,
    length_limit=100,
    length_limit_type='words',
    apply_best=False,
    alpha=0.5, # Default F1_score
    weight_factor=1.2,
    stemming=True
)


def get_summaries(directory: Path):
    eval_files = OrderedDict()
    for filename in os.listdir(directory):
        if re.search(r'D10\d\d-A\.M\.100\.\w\.\w', filename):
            with open(os.path.join(directory, filename), 'r', encoding='cp1252') as summary:
                eval_files[filename] = summary.read()
    return eval_files


def format_avg_outputs(method: int, rouge: str, averages: pd.DataFrame):
    average_scores = []
    for i, score in enumerate(averages):
        s = "{:6.5f}".format(score)
        average_scores.append(f"{method} {rouge.upper()} Average_{averages.index[i].upper()} {s}")
    return "\n".join(average_scores)


def format_eval_outputs(method: int, rouge: str, scores: pd.DataFrame):
    '''1 ROUGE-1 Eval D1001-A.M.100.A.1 R:0.07407 P:0.10465 F:0.08674'''
    rouge_scores = []
    for _, s in scores.iterrows():
        file = s['file']
        r, p, f = "{:6.5f}".format(s['r']), "{:6.5f}".format(s['p']), "{:6.5f}".format(s['f'])
        rouge_scores.append(f"{method} {rouge.upper()} Eval {file} R:{r} P:{p} F:{f}")
    return "\n".join(rouge_scores)


def get_scores(
        evalfiles: List[str], 
        summfiles: List[str]
    ):
    # set up the basefile list and assert that they are the same number
    base_eval_ids = set(map(lambda x: x[:-2], summfiles.keys()))
    base_summ_ids = set(map(lambda x: x[:-2], evalfiles.keys()))
    assert len(base_eval_ids) == len(base_summ_ids)
    assert base_eval_ids == base_summ_ids
    basefileids = sorted(list(base_summ_ids))

    # set up rouge evaluator
    evaluator_all = rouge.Rouge(apply_avg=False, **ROUGE_ARGS)

    # infer the number of methods used in summfiles
    number_of_methods = sorted(list((
        set(map(lambda x: x[-1], summfiles.keys()))
    )))

    results = []
    for method in number_of_methods:
        for base_file_id in basefileids:
            references = [v for k, v in evalfiles.items() if k.startswith(base_file_id)]
            summaries = {k: v for k, v in summfiles.items() if k.startswith(base_file_id)}
            for file, our_summary in summaries.items():
                for ref in references:
                    scores = evaluator_all.get_scores([our_summary], [ref])
                    for k, score in scores.items():
                        metrics = {k: v[0] for k, v in score[0].items()}
                        scores_with_details = dict(
                            method=method,
                            rouge=k,
                            file=file,
                            **metrics
                        )
                        results.append(scores_with_details)

    df = pd.DataFrame(results)
    df = df.sort_values(by=['rouge', 'file']).reset_index(drop=True)
    
    return df
                

def write_scores(df: pd.DataFrame, outfile: Optional[Path] = None):
    num_methods = df['method'].unique()
    rouge_scores = df['rouge'].unique()

    outf = open(outfile, 'w') if outfile else sys.stdout
    printf = lambda *args: print(*args, file=outf)

    for m, r in cartesian_product(num_methods, rouge_scores):
        printf('---------------------------------------------')
        tmp_df = df[(df['method']==m) & (df['rouge']==r)]
        averages = tmp_df[['p', 'r', 'f']].mean()
        avg_string = format_avg_outputs(m, r, averages)
        printf(avg_string)
        printf('.............................................')
        scores = format_eval_outputs(m, r, tmp_df)
        printf(scores)
    if outfile:
        outf.close()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--output_path', '-out', type=str, required=False, default=None,
        help='Path to output the ROUGE scores'
    )
    args, _ = argparser.parse_known_args()
    return args


def main():
    args = parse_args()
    evalfiles = get_summaries(EVALFILES)
    summfiles = get_summaries(SUMFILES)
    df = get_scores(evalfiles, summfiles)
    write_scores(df, args.output_path)


if __name__ == '__main__':
    main()