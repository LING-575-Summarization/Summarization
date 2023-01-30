import json
import math
"""
    Checked output by testing small test file by hand. 
    Test file can be run using:
        ./src/scripts/tf_idf.sh src/snip/tf_idf_test_data.json 1 1
"""

class TF_IDF:

    def __init__(self, docset, delta1=1, delta2=1):
        """
            Calculates the td-idf for each (term_t (lowercased), document_d) pair in the given docset_D in log_base2

            tf-idf(t, d, D) = tf(t,d)* idf(t, D)

            tf(t,d) = log(delta1 + f_td) = log[ delta1 + count(term_t in document_d) ]

            idf(t, D) = log(N / (delta2 + n_t)) + delta2 = log[ count(documents_d in docset_D) / (delta2 + count(documents d term_t appears in) ] + delta2

        """
        self.tf_idf = {}
        self.avg_tf_idf_sent_weights = {}
        self.max_tf_idf_sent_weights = {}

        self._tf = {} # (term_t, doc_d) --> tf(t,d) as float()
        self._idf = {}  # "term_t" --> idf(t, D) as float()
        self._delta1 = delta1
        self._delta2 = delta2
        self._N = len(docset)

        f_td, n_t = self._get_joint_counts(docset)
        self._build_tf_idf(f_td, n_t)
        self._calculate_tf_idf_sent_weights(docset)
    

    def _get_joint_counts(self, docset):

        f_td = {}  # (t,d) --> count(term_t in document_d)
        n_t = {}  # t --> count(documents d term_t appears in)

        for document_id, doc_data in docset.items():
            text = doc_data[3]

            doc_vocab = set()
            for paragraph in text:
                for sentence in paragraph:
                    for term in sentence:
                        term = term.lower()
                        term_doc_pair = (term, document_id)
                        if term_doc_pair not in f_td:
                            f_td[term_doc_pair] = 0
                        f_td[term_doc_pair] += 1
                        if term not in doc_vocab:
                            if term not in n_t:
                                n_t[term] = 0
                            n_t[term] += 1
                            doc_vocab.add(term)
        return f_td, n_t
    

    def _build_tf_idf(self, f_td, n_t):
        # build tf
        for term_doc_pair, count in f_td.items():
            self._tf[term_doc_pair] = math.log(self._delta1 + count, 2)

        # build idf
        for term, count in n_t.items():
            self._idf[term] = math.log(self._N / (self._delta2 + count), 2) + self._delta2

        # build tf-idf
        for term_doc_pair, term_freq in self._tf.items():
            term = term_doc_pair[0]
            cur_idf = self._idf[term]
            self.tf_idf[term_doc_pair] = term_freq * cur_idf

    
    def _calculate_tf_idf_sent_weights(self, docset):
        """ 
            Params: a sentences as a tuple of words, i.e. sent = (w1, w2, ..., wn)

            Builds dictionaries mapping (sentence, document) pairs to their respective tf-idf weights, i.e.

                avg_sent_weight(sentj) = \sum_i tf_idf(word_i, doc_d) / len(sentj)

                max_sent_weight(sentj) = \max_i tf_idf(word_i, doc_d)
            
            Returns: a dictionary (senti, doc_j) --> float
        """
        for doc_id, doc_data in docset.items():
            text = doc_data[3]
            for paragraph in text:
                sum_tf_idf = 0
                max_tf_idf = None
                for sentence in paragraph:
                    sent_repr = tuple(sentence)
                    for term in sentence:
                        cur_weight = self[term, doc_id]
                        sum_tf_idf += cur_weight
                        if max_tf_idf is None:
                            max_tf_idf = cur_weight
                        elif max_tf_idf < cur_weight:
                            max_tf_idf = cur_weight
                
                avg_tf_idf = sum_tf_idf / len(sentence)

                assert max_tf_idf is not None

                self.avg_tf_idf_sent_weights[sent_repr, doc_id] = avg_tf_idf
                self.max_tf_idf_sent_weights[sent_repr, doc_id] = max_tf_idf



    def get_tf_idf_sentence_weight(self, sentence, document: str, weight_type: str) -> float:
        """
            Returns the average tf-idf score over a whole sentence

            Params: sentence as an iterable of strings, document_id as a string, and weight_type either {"average", "max"}
        """
        sentence = tuple(sentence)
        if not ((weight_type == "average") or (weight_type == "max")):
            err_str = "weight_type parameter [" + str(weight_type) + "] must either be 'average' or 'max'"
            raise ValueError(err_str)

        sent_doc_pair = (sentence, document)
        if sent_doc_pair not in self.avg_tf_idf_sent_weights:
            return 0

        if weight_type == "max":
            return self.max_tf_idf_sent_weights[sent_doc_pair]

        return self.avg_tf_idf_sent_weights[sent_doc_pair]


    def __getitem__(self, term_doc_pair: tuple) -> float:
        """
            returns the tf-idf score for the given term, document pair. 
        """
        term, document = term_doc_pair
        term = term.lower()
        term_doc_pair = (term, document)
        if term_doc_pair not in self.tf_idf:
            return 0
        return self.tf_idf[(term, document)]


    def __str__(self):
        return str(self.tf_idf)


def create_tf_idf_dict(json_path: str, delta_1: float, delta_2: float):
    """
        Params: a file_path to a json_file, a tf smoothing delta1 (default is 1), and a idf smoothing delta2 (default is 1)
        Returns: dict[str, dict[str, TF_IDF]]

        To retrieve the tf-idf score of a given term, document pair, in a given docSetA, we can index into the TF-IDF object as follows:
                docset_tf_idf[docset_id][("term", "document")]
            or similarily:
                docset_tf_idf[docset_id]["term", "document"]
    """
    with open(json_path, "r") as final:
        read = final.read()
    docset_rep = json.loads(read)
    
    docset_tf_idf = {}

    for docset_id, docset in docset_rep.items():
        docset_tf_idf[docset_id] = TF_IDF(docset, delta_1, delta_2)

    return docset_tf_idf
