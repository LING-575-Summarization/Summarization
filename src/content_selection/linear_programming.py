import sys
import os
import pulp
import json
from collections import OrderedDict
# from tf_idf import create_tf_idf_dict
from we_tried_newtfidf import TFIDF
from nltk.util import ngrams
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.clustering import SentenceIndex, create_clusters

class LinearProgramSummarizer:

    def __init__(self,
                docset_id,
                docset,
                whole_data,
                max_sum_length,
                ngram,
                sent_length,
                delta_idf,
                delta_tf,
                eliminate_punc,
                lower_casing,
                log_base
            ):
        self.docset_id = docset_id
        self.data = docset

        # get concept weights here
        self.weights = TFIDF.idf_from_docset(
            tf_documents=docset,
            idf_documents=whole_data,
            punctuation=eliminate_punc,
            lowercase=lower_casing,
            doc_level="docset",
            ngram=ngram,
            docset_id=docset_id,
            log_tf=log,
            log_idf=log,
            delta_tf=delta_tf,
            delta_idf=delta_idf
        )

        # hyper-paraemeters
        # tfidf constraints
        self.ngram = ngram
        self.lowercase = lower_casing
        self.sent_length = sent_length
        self.L = max_sum_length

        # Initialize LP problem
        self.lp = pulp.LpProblem(name="summarizer", sense=pulp.LpMaximize)

        self.sent_decision_vars = []
        self.concept_decision_vars = []

        # Lookup tables
        # look-up list mapping z_i --> (doc_id, term_i)
        self.index_to_concept = []
        self.concept_to_index = {}
        # look-up list mapping y_j --> (doc_id, tuple(sent_j))
        self.index_to_sent = []
        self.sent_to_index = {}

        self.constraints = OrderedDict()
        self.objective_function = None

        # get objective function
        # get constraints
        self._read_data()
        self.lp.objective = self.objective_function
        self.lp.constraints = self.constraints


    def make_summary(self):
        """
            Returns a list of lists, A list of tokenized sentences
        """
        self.lp.solve()

        summary_repr = []
        for var in self.sent_decision_vars:
            if pulp.value(var) == 1:
                index = int(str(var).split("_")[1])
                sent_repr = self.index_to_sent[index][1]
                summary_repr.append(sent_repr)

        return summary_repr


    def _read_data(self):
        """
            Reads through the given docset, creating the Objective function and constraints
        """

        sent_weights = {}
        concept_weights = {}
        # A_ij set representation, if z_i is in y_j, then A_ij = 1; 0 otherwise
        # i.e. (y_j, z_i) in A_ij, A_ij = 1; 0 otherwise
        A_ij_is_1 = set()

        for doc_id, doc_data in self.data.items():
            text = doc_data[3]
            for paragraph in text:
                for sentence in paragraph:
                    # for sent_length flag
                    if len(sentence) < self.sent_length:
                        continue

                    y_j = self._make_max_length_weights(doc_id, sentence, sent_weights) # y_j
                    self._make_concept_weights(doc_id, sentence, concept_weights, y_j, A_ij_is_1) # z_n, z_n+1, ..., z_n+m

        self._concept_sentence_constraints(A_ij_is_1)
        # start making linear combinations
        # linear combination of y_j and l_j
        max_length_affine_expression = pulp.LpAffineExpression(e=sent_weights)

        # create constraint \sum_j l_j y_j <= L
        max_length_constraint = pulp.LpConstraint(
            e=max_length_affine_expression, sense=-1, name="max_length_constraint", rhs=self.L)

        self.constraints["max_length_constraint"] = max_length_constraint
        # linear combination of w_i and z_i
        self.objective_function = pulp.LpAffineExpression(e=concept_weights)


    def _concept_sentence_constraints(self, A_ij):

        cur_constraint = 0

        for z_i in self.concept_decision_vars:
            one_concept_in_sentence = {}
            for y_j in self.sent_decision_vars:

                all_concepts_in_sent = {}
                if (y_j, z_i) in A_ij:
                    all_concepts_in_sent[y_j] = 1
                    one_concept_in_sentence[y_j] = 1
                else:
                    all_concepts_in_sent[y_j] = 0
                    one_concept_in_sentence[y_j] = 0
                all_concepts_in_sent[z_i] = -1
                z_i_in_y_j_linear_combo = pulp.LpAffineExpression(e=all_concepts_in_sent)
                cur_name = "c_" + str(cur_constraint)
                z_i_in_y_j = pulp.LpConstraint(e=z_i_in_y_j_linear_combo, sense=-1, name=cur_name, rhs=0)
                self.constraints[cur_name] = z_i_in_y_j
                cur_constraint += 1
            one_concept_in_sentence[z_i] = -1
            cur_name = "d_" + str(cur_constraint)
            one_zi_linear_combo = pulp.LpAffineExpression(e=one_concept_in_sentence)
            one_zi = pulp.LpConstraint(e=one_zi_linear_combo, sense=1, name=cur_name, rhs=0)
            self.constraints[cur_name] = one_zi


    def _make_max_length_weights(self, doc_id, sentence, sent_weights):
        cur_index = len(self.index_to_sent)
        y_j_name = "y_" + str(cur_index)
        y_j = pulp.LpVariable(name=y_j_name, cat='Binary')
        self.sent_decision_vars.append(y_j)
        sent_weights[y_j] = len(sentence)

        doc_id_sent = (doc_id, tuple(sentence))
        self.index_to_sent.append(doc_id_sent)
        self.sent_to_index[doc_id_sent] = cur_index

        return y_j


    def _make_concept_weights(self, doc_id, sentence, concept_weights, y_j, A_ij_is_1):
        sentence = list(ngrams(sentence, self.ngram, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))

        for term in sentence:

            # for lowercasing flag
            if self.lowercase:
                term = tuple([word.lower() for word in term])

            doc_id_concept = (doc_id, term)

            if doc_id_concept not in self.index_to_concept:
                # initialize each z_i for each term
                cur_index = len(self.index_to_concept)
                z_i_name = "z_" + str(cur_index)
                z_i = pulp.LpVariable(name=z_i_name, cat='Binary')
                self.concept_decision_vars.append(z_i)

                # map (doc_id, term) --> i
                self.concept_to_index[doc_id_concept] = cur_index
                # map i --> (doc_id, term)
                # append (doc_id, term) to lookup table concepts
                self.index_to_concept.append(doc_id_concept)

                concept_weights[z_i] = self.weights[term, self.docset_id]
            else:  # need to look up z_i in lookup table
                cur_index = self.concept_to_index[doc_id_concept]
                z_i = self.concept_decision_vars[cur_index]

            A_ij_is_1.add((y_j, z_i))


def read_json(json_path):
    with open(json_path, "r") as final:
        docset_rep = json.load(final)
    return docset_rep


def detokenize_summary(summary):
    detokenizer = TreebankWordDetokenizer()
    summary_str = ""
    for sent in summary:
        summary_str += detokenizer.detokenize(sent) + "\n"
    return summary_str


if __name__ == '__main__':
    # from shell script
    json_path = sys.argv[1]
    max_summary_length = int(sys.argv[2])
    output_dir = sys.argv[3]
    ngram = int(sys.argv[4])
    sent_length = int(sys.argv[5])
    delta_idf = float(sys.argv[6])
    delta_tf = float(sys.argv[7])
    eliminate_punc = bool(sys.argv[8])
    lower_casing = bool(sys.argv[9])
    log = bool(sys.argv[10])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    docset_rep = read_json(json_path)
    fractional_order = SentenceIndex(json_path)

    # create idf_docsset for tf-idf object
    idf_docset = {}
    for docset_id, docset in docset_rep.items():
        for document, data in docset.items():
            doc_key = docset_id + "." + document
            idf_docset[doc_key] = data

    iter = 0
    for docset_id, docset in docset_rep.items():
        model = LinearProgramSummarizer(
            docset_id,
            docset,
            idf_docset,
            max_summary_length,
            ngram,
            sent_length,
            delta_idf,
            delta_tf,
            eliminate_punc,
            lower_casing,
            log
        )
        # gets unordered summary
        summary = model.make_summary()

        # order the summary
        summary = create_clusters(docset_id, summary, fractional_order, json_path)
        summary = detokenize_summary(summary)

        # output summaries to correct file
        docset_id = docset_id.split("-")[0]
        id_part1 = docset_id[:-1]
        id_part2 = docset_id[-1]
        output_summary_file = output_dir + "/" + id_part1 + "-A.M.100." + id_part2 + ".1"
        with open(output_summary_file, 'w') as output:
            output.write(summary)

        iter += 1
        if iter == 2:
            break
