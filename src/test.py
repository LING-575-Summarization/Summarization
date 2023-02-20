from vectorizer import *
from tqdm import tqdm
import json, os
from content_selection.lexrank import LexRankFactory

def main():
    with open('data/devtest.json', 'r') as datafile:
        data = json.load(datafile).keys()
    for i, vector in enumerate(['tfidf', 'word2vec', 'bert']):
        LexRank = LexRankFactory(vector)
        for docset_id in tqdm(data):
            lx = LexRank.from_data(docset_id, 'data/devtest.json', 
                sentences_are_documents=True,  min_length=5, min_jaccard_dist=0.6)
            result = lx.obtain_summary(detokenize=True)
            spl = str(docset_id).split("-", maxsplit=1)
            id0, id1 = spl[0], spl[1]
            id0 = id0[:-1]
            output_file = os.path.join('outputs', 'D4', f'{id0}-A.M.100.{id1}.{i+1}')
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