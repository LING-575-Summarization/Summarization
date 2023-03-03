'''
More robust DocumentToBert model functions
e.g., can now pretrain on the training data
'''

from typing import *
from pathlib import Path
from transformers import DocumentToBertModel, DocumentToBertTokenizerFast, Trainer
from datasets import Dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils import dataset_loader, get_summaries

DETOKENIZER = TreebankWordDetokenizer()


class TransformerModel:

    tokenizer = DocumentToBertTokenizerFast.from_pretrained(
        "DocumentToBert-base-uncased"
    )
    
    model =  DocumentToBertModel.from_pretrained(
        "DocumentToBert-base-uncased",
        output_hidden_states=True
    )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)
    

    def pretrain(pretrain_data: Path, summary_data: Path):
        data = dataset_loader(pretrain_data, return_dict=True)
        docset_doc_ids = list(data.keys())
        docsets = list(map(lambda x: x[:-2], docset_doc_ids))
        document_text = list(map(data.values(), lambda x: DETOKENIZER.detokenize(x)))
        summaries = get_summaries(summary_data)
        pretrain_ds = Dataset.from_dict({
            "document_set": docsets,
            "document_id": docset_doc_ids, 
            "document": document_text
        })
        

