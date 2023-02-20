from vectorizer import *
from tqdm import tqdm
import json, os
from content_selection.lexrank import LexRankFactory
from dataclasses import dataclass

@dataclass
class Experiment:
    idf_level: str
    ngram: int
    delta_idf: float
    log_tf: bool

    def as_dict(self):
        x = self.__dict__
        m = x.pop('idf_level')
        l = False if m == "sentence" else True
        return x, l

EXPERIMENTS = [
    Experiment("sentence", 1, 0., False),
    Experiment("documset", 1, 0., False),
    Experiment("sentence", 2, 0., False),
    Experiment("documset", 2, 0., False),
    Experiment("sentence", 1, 0.7, False), 
    Experiment("documset", 1, 0.7, False), 
    Experiment("sentence", 1, 0., True), 
    Experiment("documset", 1, 0., True), 
]


def main():
    with open('data/devtest.json', 'r') as datafile:
        data = json.load(datafile).keys()
    LexRank = LexRankFactory('tfidf')
    for i, expt in enumerate(EXPERIMENTS):
        args, idf_docset = expt.as_dict()
        for docset_id in tqdm(data):
            if idf_docset:
                lx = LexRank.from_data(datafile='data/devtest.json', eval_docset=docset_id,
                    sentences_are_documents=True, min_length=5, min_jaccard_dist=0.6, **args)
            else:
                lx = LexRank.from_data(datafile='data/devtest.json', documentset=docset_id,
                    sentences_are_documents=True, min_length=5, min_jaccard_dist=0.6, **args)
            result = lx.obtain_summary(detokenize=True)
            id0 = docset_id[0:5]
            id1 = docset_id[-3]
            output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i+1}')
            with open(output_file, 'w') as outfile:
                outfile.write(result)

if __name__ == '__main__':
    main()
    # print("Testing DocumentToTFIDF document-level")
    # x = DocumentToTFIDF.from_data('D1001A-A', 'data/devtest.json')
    # print(x.similarity_matrix())
    # print("Testing DocumentToTFIDF sentence-level")
    # x = DocumentToTFIDF.from_data('D1001A-A', 'data/devtest.json', sentences_are_documents=True)
    # print(x.similarity_matrix())

    # eval_docs_s, _ = docset_loader('D1001A-A', 'data/devtest.json', sentences_are_documents=True)
    # eval_docs_d, _ = docset_loader('D1001A-A', 'data/devtest.json')
    # print("Testing DocumentToTFIDF w/ eval_docs (doc-level)")
    # x = DocumentToTFIDF.from_data('D1001A-A', 'data/devtest.json', eval_documents=eval_docs_d)
    # print(x.similarity_matrix())
    # print("Testing DocumentToTFIDF w/ eval_docs (sentence-level)")
    # x = DocumentToTFIDF.from_data('D1001A-A', 'data/devtest.json', 
    #                               sentences_are_documents=True, eval_documents=eval_docs_s)
    # print(x.similarity_matrix())

    # print("Testing DocumentToDistilBert document-level")
    # x = DocumentToDistilBert.from_data('D1001A-A', 'data/devtest.json')
    # print(x.similarity_matrix())
    # print("Testing DocumentToDistilBert sentence-level")
    # x = DocumentToDistilBert.from_data('D1001A-A', 'data/devtest.json', sentences_are_documents=True)
    # print(x.similarity_matrix())

    # print("Testing DocumentToWord2Vec document-level")
    # x = DocumentToWord2Vec.from_data('D1001A-A', 'data/devtest.json')
    # print(x.similarity_matrix())
    # print("Testing DocumentToWord2Vec sentence-level")
    # x = DocumentToWord2Vec.from_data('D1001A-A', 'data/devtest.json', sentences_are_documents=True)
    # print(x.similarity_matrix())