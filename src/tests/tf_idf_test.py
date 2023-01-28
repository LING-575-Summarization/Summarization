import sys
import json
from tf_idf import create_tf_idf_dict

def test_tf_idf(docset_rep, tf_idf_repr):
    for docset_id, docset in docset_rep.items():
        cur_tf_idf = tf_idf_repr[docset_id]
        print(cur_tf_idf.tf_idf)
        print()
        print("max", cur_tf_idf.max_tf_idf_sent_weights)
        print("avg", cur_tf_idf.avg_tf_idf_sent_weights)
        print("##########\n")
        for doc_id, doc_data in docset.items():
            for para in doc_data[3]:
                for sentence in para:
                    sent_repr = tuple(sentence)
                    print("avg", cur_tf_idf.get_tf_idf_sentence_weight(sent_repr, doc_id, "average"))
                    print("max", cur_tf_idf.get_tf_idf_sentence_weight(sent_repr, doc_id, "max"))
                    print()
                    cur_tf_idf.get_tf_idf_sentence_weight(sent_repr, doc_id, "should error")


if __name__ == '__main__':
    ## for testing purposes
    json_path = sys.argv[1]
    delta1 = float(sys.argv[2])
    delta2 = float(sys.argv[3])

    with open(json_path, "r") as final:
        read = final.read()
    docset_rep = json.loads(read)

    tf_idf_repr = create_tf_idf_dict(json_path, delta1, delta2)
    test_tf_idf(docset_rep, tf_idf_repr)