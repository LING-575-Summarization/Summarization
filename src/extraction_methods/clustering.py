"""
    This is an information ordering algorithm, based on clustering (Topic Modelling).

    To run:
        1) Create a SentenceIndex object BEFORE writing out any summaries (this goes through the whole json file,
            to map sentence to their corresponding fractional ordering)
        2) Call create_clusters(docset_id, summary, fractional_order) to cluster and return
            the ordered summary as a list of tokenized sentences

            Example (from LinearProgramming):

            1)
                json_path = "outputs/devtest.json"
                fractional_order = SentenceIndex(json_path)

                for docset_id, docset in json_file:
                    summary = LinearProgramming(docset_id, docset)

                    # note: will return a list of lists (list of tokenized sentences)
            2)      ordered_summary = create_clusters(docset_id, summary, fractional_order)

"""

import sys
import json
import re
from statistics import median
from utils import flatten_list, docset_loader
from sklearn.cluster import KMeans
from vectorizer import TFIDFModel, Word2VecModel, BERTModel


class SentenceIndex:


    def __init__(self, data_file_path):
        """
            Creates fractional ordering of sentences, e.g.
                sent1 is seen at position 3 out of 7 sentences in the whole document, this returns 3/7
            Note: If the same sentence appears more than once in the docset, then we return the
                one with the lowest fractional ordering

            Params:
                - stage := {"training", "devtest", "test"}
        """
        self.data = self._read_json(data_file_path)
        self.sent_to_fractional_ordering = {}  # (docset_id, str) --> index
        self.docset_id_to_sentences = {}  # (docset_id) --> str

        self._get_indices()


    def _read_json(self, json_path):
        with open(json_path, "r") as final:
            read = final.read()
        docset_rep = json.loads(read)
        return docset_rep


    def _get_indices(self):
        for docset_id, docset in self.data.items():
            self.docset_id_to_sentences[docset_id] = []
            for document_id, document in docset.items():

                text = document[-1]
                # flatten paragraphs
                text = flatten_list(text)

                # get total number of sentences in current document
                num_sentences = len(text)
                for index, sentence in enumerate(text):
                    self.docset_id_to_sentences[docset_id].append(sentence)
                    sentence = " ".join(sentence)
                    docset_id_sent = (docset_id, sentence)
                    fractional_ordering = index / num_sentences

                    # updates fractional ordering for given docset, sentence pair
                    if (docset_id_sent not in self.sent_to_fractional_ordering) \
                            or self.sent_to_fractional_ordering[docset_id_sent] > fractional_ordering:
                        self.sent_to_fractional_ordering[docset_id_sent] = fractional_ordering


    def get(self, docsetid_sentence):
        """
            Params:
                - docset_id: A string
                - sentence: Either a tokenized string separated by whitespace,
                    or an iterable representing a tokenized string
        """
        if len(docsetid_sentence) != 2:
            raise ValueError(f"argument is not a docset_id, sentence pair, but is {docsetid_sentence}")
        docset_id, sentence = docsetid_sentence
        if isinstance(sentence, list) or isinstance(sentence, tuple):
            sentence = " ".join(sentence)
        elif isinstance(sentence, str):
            sentence = sentence.strip()
        else:
            raise ValueError(
                f"sentence argument not tokenized string separated by whitespace, "
                f"or an iterable, but is {type(sentence)}"
            )

        return self.sent_to_fractional_ordering[(docset_id, sentence)]


    def __getitem__(self, docsetid_sentence):
        """
            Params:
                - docset_id: A string
                - sentence: Either a tokenized string separated by whitespace,
                    or an iterable representing a tokenized string
        """
        return self.get(docsetid_sentence)


    def __str__(self):
        return str(self.sent_to_fractional_ordering)


class Clustering:

    def __init__(self, docset_id, sentence_indices, json_path):
        """
            Params:
                - summary: a list of sentences comprising a summary
                - docset_id: a string
                - sentence_indices: A SentenceIndex Object for fast lookup,
                    mapping sentences to indices and mapping docset_id to sentences
        """
        self.docset_id = docset_id
        self.json_path = json_path
        self.sentence_indices = sentence_indices
        self.embedding_to_sentence = {}

        self.cluster_to_sentences = {}
        self.sentence_to_cluster = {}

        self._get_sentence_embeddings(self.sentence_indices.docset_id_to_sentences[self.docset_id])
        self.cluster_docset()
        self.ordered_clusters = self.order_cluster_themes()


    def _get_sentence_embeddings(self, sentences):
        # random.seed(11)

        # TFIDF
        docset, indices = docset_loader(self.docset_id, self.json_path)
        model = TFIDFModel(docset)

        # Word2Vec
        # model = Word2VecModel()

        # BERTModel
        # model = BERTModel()

        for sentence in sentences:
            # create random embeddings for testing purposes
            # embedding = tuple(random.choices(range(10), k=10))
            # embedding = Hilly's embedding
            sent_str = " ".join(sentence)
            if not re.search(r'\w', sent_str):
                continue
            embedding = model.vectorize_sentence(sent_str)
            embedding = tuple(embedding)
            if embedding not in self.embedding_to_sentence:
                self.embedding_to_sentence[embedding] = set()
            # embedding -> set of sentences since many have the same embedding
            self.embedding_to_sentence[embedding].add(tuple(sentence))


    def cluster_docset(self):
        embeddings = list(self.embedding_to_sentence.keys())

        # default values for testing purposes
        # note: algorithm='auto' is the same thing as algorithm='lloyd' (use "auto" if doesn't work)
        kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')

        clusters = kmeans.fit_predict(embeddings)

        for index, cluster_index in enumerate(clusters):
            sentences = self.embedding_to_sentence[embeddings[index]]

            if cluster_index not in self.cluster_to_sentences:
                self.cluster_to_sentences[cluster_index] = set()
            for sentence in sentences:
                sentence = tuple(sentence)
                self.cluster_to_sentences[cluster_index].add(sentence)
                self.sentence_to_cluster[sentence] = cluster_index
            
    
    def order_cluster_themes(self):
        cluster_to_frac_ordering = dict()

        for cluster, sentences in self.cluster_to_sentences.items():
            cur_sum = 0

            # MEAN
            # for sentence in sentences:
            #     cur_sum += self.sentence_indices[self.docset_id, sentence]
            # avg_fractional_ordering = cur_sum / len(sentences)
            # cluster_to_frac_ordering[cluster] = avg_fractional_ordering

            # MEDIAN
            indices = []
            for sentence in sentences:
                idx = self.sentence_indices[self.docset_id, sentence]
                indices.append(idx)
            med = median(indices)
            cluster_to_frac_ordering[cluster] = med

        # sort from least to greatest fractional ordering
        ordered_clusters = sorted(cluster_to_frac_ordering.items(), key=lambda item: item[1])

        return ordered_clusters


    def order_summary(self, unordered_summary):
        """
            Params:
                - A list of lists
                    A list of sentences, where each sentence is a list of tokens
        """
        ordered_summary = []
        for cluster_index, fractional_ordering in self.ordered_clusters:
            block = []
            for sentence in unordered_summary:
                # for testing purposes
                # sentence = sentence[0].strip().split()

                sentence = tuple(sentence)
                if sentence in self.cluster_to_sentences[cluster_index]:
                    block.append(sentence)
            ordered_block = self._order_block(block)
            ordered_summary.extend(ordered_block)
        return ordered_summary


    def _order_block(self, block):
        block_ordering = {}
        for sentence in block:
            fractional_ordering = self.sentence_indices[self.docset_id, sentence]
            block_ordering[sentence] = fractional_ordering

        sorted_block = dict(sorted(block_ordering.items(), key=lambda item: item[1])).keys()
        sorted_block = [list(sentence) for sentence in sorted_block]

        return sorted_block
                

def create_clusters(docset_id, summary, sent_indices, json_path):
    """
        Note: Requires a pre-loaded SentenceIndex object, please load this SenteneIndex
            just once to avoid loading the given json file more than once
        Params:
            - docset_id: A string representing the docset_id
            - summary: A list of lists
                    A list of sentences, where each sentence is a list of tokens
            - sent_indices: A SentenceIndex Object
        Returns:
            - An ordered summary
            - An ordered list of sentences, where each sentence is a list of tokens
    """
    clusters = Clustering(docset_id, sent_indices, json_path)
    summary = clusters.order_summary(summary)
    return summary


if __name__ == '__main__':
    pass


# if __name__ == '__main__':
#     docset_id = sys.argv[1]
#     data_file_path = sys.argv[2]
#
#     sent_indices = SentenceIndex(data_file_path)
#
#     Note: should be a list of lists, where the inner list is a list of tokens, not a list of a whole sentence string
#     summary = [["At one point , two bomb squad trucks sped to the school after a backpack scare ."],
#             ["Phone : ( 888 ) 603-1036"],
#             ["Please comfort this town . ''"],
#             ["Many looked for it Saturday morning on top of Mt ."],
#             ["But what community was it from ?"],
#             ["There are the communities that existed already , like Columbine students and Columbine Valley residents ."],
#             ["Brothers Jonathan and Stephen Cohen sang a tribute they wrote ."],
#             ["`` Columbine ! ''"],
#             ["`` Love is stronger than death . ''"],
#             ["Some players said the donations and support will encourage them to play better ."]]
#
#     create_clusters(docset_id, summary, sent_indices)



    # test_sentence = "But Simmons ' attorney , Seth Waxman , replied that the " \
    #                 "death penalty was a useless deterrent against minors since `` " \
    #                 "they weigh risks differently '' to adults ."
    # print(sent_indices)
    # 0.9655172413793104
    # print(sent_indices['D0944H-A', test_sentence])
    # print(sent_indices[test_sentence])