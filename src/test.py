from vectorizer import *
from content_selection.lexrank import LexRankFactory

def main():
    NewClass = LexRankFactory('bert')
    n = NewClass.from_data('D1001A-A', 'data/devtest.json', sentences_are_documents=True, min_length=5)
    print(n.obtain_summary(0.2, error=1e-32))
    n = NewClass.from_data('D1003A-A', 'data/devtest.json', sentences_are_documents=True, min_length=5)
    print(n.obtain_summary(0.2, error=1e-32))

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