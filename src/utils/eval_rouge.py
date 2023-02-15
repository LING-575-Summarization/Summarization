import rouge
from typing import List, Optional
from pathlib import Path
from collections import OrderedDict
import sys, os


user_root = os.path.expanduser("~")
EVALFILES = os.path.join(user_root, "/dropbox/22-23/575x/Data/models/devtest/")
SUMFILES = os.path.join(user_root, "575-Summarization", "outputs", "D4")


def get_summaries(directory: Path):
    eval_files = OrderedDict()
    for filename in os.listdir(directory):
        print(filename)
        with open(filename, 'r') as summary:
            eval_files[filename] = summary.read()
    return eval_files


def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def get_scores(
        textlist: List[str], 
        summlist: List[str],
        outfile: Optional[Path] = None
    ):
    assert len(textlist) == len(summlist), "textlist and summlist should have same length"
    
    
    for aggregator in ['Avg', 'Individual']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'

        evaluator = rouge.Rouge(metrics=['rouge-n'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=apply_avg,
                            apply_best=apply_best,
                            alpha=0.5, # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

        # overwrite file
        if outfile:
            with open(outfile, 'w') as fp:
                pass

        out = open(outfile, 'a') if outfile else sys.stdout
        for generated_sum, standard_sum in zip(textlist, summlist):
            scores = evaluator.score([generated_sum], [standard_sum])
            print(scores, file=out)
        if outfile:
            out.close()


def main():
    textlist = get_summaries(EVALFILES)
    summlist = get_summaries(SUMFILES)
    get_scores(textlist, summlist)


if __name__ == '__main__':
    main()