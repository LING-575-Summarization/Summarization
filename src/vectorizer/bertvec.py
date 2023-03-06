if __name__ == '__main__':
    import sys, os
    module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(module_path)
    from vector_api import VectorModel, DocumentToVectors
else:
    from .vector_api import VectorModel, DocumentToVectors

from transformers import (AutoModelForMaskedLM, AutoTokenizer, logging, Trainer, 
                          TrainingArguments, DataCollatorForLanguageModeling)
from datasets import Dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils import dataset_loader    
import torch
import numpy as np
from pathlib import Path
from typing import *

logging.set_verbosity_warning() 

'''
Citations:
- Huggingface (https://huggingface.co/docs/transformers/model_doc/bert)
'''

DETOKENIZER = TreebankWordDetokenizer()

TRAINING_ARGS = TrainingArguments(
    output_dir="saved_models",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=12
)

class BERTModel(VectorModel):
    def __init__(self, model: AutoModelForMaskedLM = None, tokenizer: AutoTokenizer = None) -> None:
        '''
        Instantiate a BERT model and tokenizer to obtain the [CLS] token from the sentence
        a reduction method to turn a sentence into a single vector (default is centroid)
        Also instantiates a Fasttext model to account for unseen vocabulary items
        '''
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        else:
            self.tokenizer = tokenizer
        if model is None:
            self.model = AutoModelForMaskedLM.from_pretrained('sentence-transformers/all-distilroberta-v1', output_hidden_states=True)
        else:
            self.model = model

        self.max_length = self.model.config.max_length
    

class DocumentToBert(DocumentToVectors, BERTModel):
    def __init__(
            self, 
            documents: List[List[str]], 
            indices: Dict[str, Union[np.array, List[float]]], 
            **kwargs
        ) -> None:
        pretrain_ds = kwargs.pop('pretrain', None)
        if pretrain_ds:
            model, tokenizer = self.pretrain(pretrain_ds)
            kwargs['model'] = model
            kwargs['tokenizer'] = tokenizer
        super().__init__(documents, indices, **kwargs)
    

if __name__ == '__main__':
    bert = DocumentToBert.from_data(datafile='data/devtest.json')
    print(bert.similarity_matrix())