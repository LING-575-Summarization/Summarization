from document_set_loader import get_summaries


PEER_ROOT = 'repo/outputs/D4-lexrank'
MODEL_ROOT = 'eval/devtest'


def main():
    '''Makes a ROUGE XML file'''

    outxml = open('rouge/rouge_run.xml', 'w')

    evalfiles = get_summaries(MODEL_ROOT)
    summfiles = get_summaries(PEER_ROOT)

    l = len("D1046-A.M.100.H.5")

    base_eval_ids = set(map(lambda x: x[:-2-(len(x)-l)], summfiles.keys()))
    base_summ_ids = set(map(lambda x: x[:-2], evalfiles.keys()))

    assert len(base_eval_ids) == len(base_summ_ids)
    assert base_eval_ids == base_summ_ids
    basefileids = sorted(list(base_summ_ids))

    # infer the number of methods used in summfiles
    number_of_methods = sorted(list((
        set(map(lambda x: x.split('.')[-1], summfiles.keys()))
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
