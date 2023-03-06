if __name__ == '__main__':
    import sys, os
    module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(module_path)
    from vector_api import VectorModel, DocumentToVectors
else:
    from .vector_api import VectorModel, DocumentToVectors

from transformers import AutoModel, AutoTokenizer, logging
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
        '''
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2', output_hidden_states=True)
        self.max_length = self.model.config.max_length
        

    def vectorize_sentence(self, sentence: List[str]) -> np.ndarray:
        '''
        Return a vector representation of the sentece
        Also truncates sentences that are too long
        Args:
            - sentence: a tokenized list of words
        Returns:
            - 1 dimensional np.ndarray of floats
        CITATION: https://www.sbert.net/index.html
        ("Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks")
        '''
        sentence_as_string = DETOKENIZER.detokenize(sentence)
        tokenized_sentence = self.tokenizer(
            sentence_as_string, 
            return_tensors='pt', 
            truncation=True, 
            padding=True,
            add_special_tokens=True,
            return_attention_mask=True
        )
        with torch.no_grad():
            outputs = self.model(**tokenized_sentence)
            pooled_sentence = outputs[0]
            attention_masks = (tokenized_sentence['attention_mask']
                                    .unsqueeze(-1)
                                    .expand(pooled_sentence.size()))
            pooled_sentence = torch.sum(pooled_sentence * attention_masks, axis=1)/torch.clamp(attention_masks.sum(1), min=1e-9)
            pooled_sentence_N = torch.nn.functional.normalize(pooled_sentence, p=2, dim=1)
        sentence_embedding = pooled_sentence_N.squeeze().numpy()
        return sentence_embedding
    

class DocumentToBert(DocumentToVectors, BERTModel):
    pass
    

if __name__ == '__main__':
    bert = DocumentToBert.from_data(datafile='data/devtest.json')
    print(bert.similarity_matrix())