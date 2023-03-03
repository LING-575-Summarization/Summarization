from .vector_api import VectorModel, DocumentToVectors
from transformers import BertModel, BertTokenizerFast, logging
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import numpy as np
from typing import *

logging.set_verbosity_warning() 

'''
Citations:
- Huggingface (https://huggingface.co/docs/transformers/model_doc/bert)
'''

DETOKENIZER = TreebankWordDetokenizer()

class BERTModel(VectorModel):
    def __init__(self) -> None:
        '''
        Instantiate a BERT model and tokenizer to obtain the [CLS] token from the sentence
        a reduction method to turn a sentence into a single vector (default is centroid)
        Also instantiates a Fasttext model to account for unseen vocabulary items
        '''
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )
        self.model = BERTModel.from_pretrained(
            "base-base-uncased",
            output_hidden_states=True
        )
        self.max_length = self.model.config.max_length
        

    def vectorize_sentence(self, sentence: List[str]) -> np.ndarray:
        '''
        Return a vector representation of the sentece
        Also truncates sentences that are too long
        Args:
            - sentence: a tokenized list of words
        Returns:
            - 1 dimensional np.ndarray of floats
        '''
        sentence_as_string = DETOKENIZER.detokenize(sentence)
        tokenized_sentence = self.tokenizer(
            sentence_as_string, 
            return_tensors='pt', 
            max_length=self.max_length,
            truncation=True, 
            add_special_tokens=True
        )
        with torch.no_grad():
            pooled_sentence = self.model(**tokenized_sentence).pooler_output
        cls_token = np.mean(pooled_sentence.numpy(), axis=-1)
        return cls_token
    

class DocumentToBert(DocumentToVectors, BERTModel):
    pass
    

if __name__ == '__main__':
    x = DocumentToBert.from_data('D1001A-A', 'data/devtest.json')
    print(x.similarity_matrix())