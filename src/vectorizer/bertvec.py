if __name__ == '__main__':
    import sys, os
    module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(module_path)

from vector_api import VectorModel, DocumentToVectors
from transformers import (AutoModelForMaskedLM, AutoTokenizer, RobertaModel, logging, Trainer, 
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
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = tokenizer
        if model is None:
            self.model = AutoModelForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True)
        else:
            self.model = model

        self.max_length = self.model.config.max_length


    def pretrain(self, pretrain_data: Path):
        model = AutoModelForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        block_size = 128

        data, _ = dataset_loader(pretrain_data, return_dict=True, sentences_are_documents=True)
        docset_doc_ids = list(data.keys())
        docsets = list(map(lambda x: x[:-2], docset_doc_ids))
        document_text = list(map(lambda x: DETOKENIZER.detokenize(x), data.values()))
        pretrain_ds = Dataset.from_dict({"document_set": docsets, "document_id": docset_doc_ids, 
                                         "document": document_text})
        
        def tokenize(examples):
            return tokenizer(examples['document'], truncation=True)
        
        def group_text(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        group_texts = pretrain_ds.map(tokenize, batched=True, num_proc=4,
                                      remove_columns=pretrain_ds.column_names)
        tokenized_ds = group_texts.map(group_text, batched=True, num_proc=4)
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        trainer = Trainer(
            model=model, 
            args=TRAINING_ARGS, 
            train_dataset=tokenized_ds,
            eval_dataset=tokenized_ds, 
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model()

        # model = AutoModelForMaskedLM.from_pretrained('saved_models')
        
        # tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        return model, tokenizer
        

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
            add_special_tokens=True,
            return_attention_mask=True
        )
        with torch.no_grad():
            outputs = self.model(**tokenized_sentence)
            pooled_sentence = outputs.hidden_states[-4:-1]
            attention_masks = (tokenized_sentence['attention_mask']
                                    .squeeze()
                                    .unsqueeze(-1)
                                    .expand(pooled_sentence.size()))
            pooled_sentence = torch.sum(pooled_sentence, axis=0)/torch.sum(attention_masks, axis=0)
        cls_token = pooled_sentence.numpy()
        return cls_token
    

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
    bert = DocumentToBert.from_data(datafile='data/devtest.json', 
                                    documentset='D1001A-A', 
                                    pretrain='data/training.json')
    print(bert.similarity_matrix())