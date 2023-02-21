import os
import re
from pathlib import Path
from collections import OrderedDict

PEER_ROOT = 'repo/outputs/D4'
MODEL_ROOT = '/dropbox/22-23/575x/Data/models/devtest/'


def get_summaries(directory: Path):
    eval_files = OrderedDict()
    for filename in os.listdir(directory):
        if re.search(r'D10\d\d-A\.M\.100\.\w\.\w', filename):
            with open(os.path.join(directory, filename), 'r', encoding='cp1252') as summary:
                eval_files[filename] = summary.read()
    return eval_files


def main():
    '''Makes a ROUGE XML file'''

    outxml = open('rouge/rouge_run.xml', 'w')

    evalfiles = get_summaries(MODEL_ROOT)
    summfiles = get_summaries(PEER_ROOT)

    base_eval_ids = set(map(lambda x: x[:-2], summfiles.keys()))
    base_summ_ids = set(map(lambda x: x[:-2], evalfiles.keys()))

    print(base_eval_ids)
    print(base_summ_ids)

    assert len(base_eval_ids) == len(base_summ_ids)
    assert base_eval_ids == base_summ_ids
    basefileids = sorted(list(base_summ_ids))

    # infer the number of methods used in summfiles
    number_of_methods = sorted(list((
        set(map(lambda x: x[-1], summfiles.keys()))
    )))

    print(f"<ROUGE_EVAL version=\"1.5.5\">", file=outxml)

    for base_file_id in basefileids:
        print(f"<EVAL ID=\"{base_file_id}\">", file=outxml)
        print(f"<PEER-ROOT>", file=outxml)
        print(f"{PEER_ROOT}", file=outxml)
        print(f"</PEER-ROOT>", file=outxml)
        print(f"<MODEL-ROOT>", file=outxml)
        print(f"{MODEL_ROOT}", file=outxml)
        print(f"</MODEL-ROOT>", file=outxml)
        print(f"<INPUT-FORMAT TYPE=\"SPL\">", file=outxml)
        print(f"</INPUT-FORMAT>", file=outxml)
        print(f"<PEERS>", file=outxml)
        for method in number_of_methods:
            print(f"<P ID=\"{method}\">{base_file_id}.{method}</P>", file=outxml)
        print(f"</PEERS>", file=outxml)
        references = [k for k in evalfiles.keys() if k.startswith(base_file_id)]
        print(f"<MODELS>", file=outxml)
        for model in references:
            print(f"<M ID=\"{model[-1]}\">{model}</M>", file=outxml)
        print(f"</MODELS>", file=outxml)
        print(f"</EVAL>", file=outxml)

    print(f"</ROUGE_EVAL>", file=outxml)

    outxml.close()

if __name__ == '__main__':
    main()