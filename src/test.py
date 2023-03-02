from vectorizer import *
from tqdm import tqdm
import json, os
from utils import docset_loader, dataset_loader, get_summaries
from content_selection.lexrank import LexRankFactory
from dataclasses import dataclass
from content_selection.lexrank_estimator import LexrankClassifier

@dataclass
class Experiment:
    idf_level: str
    ngram: int
    delta_idf: float
    log_tf: bool
    ignore_stopwords: bool
    min_length: int = 7
    min_jaccard_dist: float = 0.6

    def as_dict(self):
        x = self.__dict__
        m = x.pop('idf_level')
        l = False if m == "sentence" else True
        return x, l

EXPERIMENTS = [
    Experiment("sentence", 1, 0., True), 
    Experiment("documset", 1, 0., True), 
]

def main():
    with open('data/devtest.json', 'r') as datafile:
        data = json.load(datafile).keys()
    LexRank = LexRankFactory('tfidf')
    for i, expt in enumerate(EXPERIMENTS):
        args, idf_docset = expt.as_dict()
        if idf_docset:
            lx = LexRank.from_data(datafile='data/devtest.json', sentences_are_documents=True, **args)
            for docset_id in tqdm(data, desc="Evaluating documents"):
                docset, indices = docset_loader(
                    'data/devtest.json', docset_id, sentences_are_documents=True)
                lx.replace_evaldocs(docset, indices)
                result = lx.obtain_summary(detokenize=True)
                id0 = docset_id[0:5]
                id1 = docset_id[-3]
                output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i+1}')
                with open(output_file, 'w') as outfile:
                    outfile.write(result)
        else:
            for docset_id in tqdm(data, desc="Evaluating documents"):
                lx = LexRank.from_data(datafile='data/devtest.json', documentset=docset_id,
                    sentences_are_documents=True, **args)
                result = lx.obtain_summary(detokenize=True)
                id0 = docset_id[0:5]
                id1 = docset_id[-3]
                output_file = os.path.join('outputs', 'D4-lexrank', f'{id0}-A.M.100.{id1}.{i+1}')
                with open(output_file, 'w') as outfile:
                    outfile.write(result)
            

def scikit():
    with open('data/devtest.json', 'r') as datafile:
        data = json.load(datafile).keys()
    dataset, indices = dataset_loader('data/devtest.json', sentences_are_documents=True)
    lx = LexrankClassifier('tfidf')
    lx = lx.fit((dataset, indices))
    for docset_id in tqdm(data, desc="Evaluating documents"):
        dataset, indices = docset_loader('data/devtest.json', docset_id, sentences_are_documents=True)
        summaries = get_summaries('eval/devtest', docset_id)
        print(lx.score((dataset, indices), summaries))


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