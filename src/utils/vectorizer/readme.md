# Vectorizor module explained

This module contains several classes that, when instantiated, can be used generate vectors.

> **Note**: This module requires several packages including PyTorch, Transformers, and Gensim.
> Please install these packages to your virtual environment.

## Types of classes

There are two types of classes:

1. [Vector]Model
2. DocumentTo[Vector]

Where, [Vector] is replaced by the method you'd like to use to generate the vectors. These methods are:

1. TFIDF
2. Word2Vec
3. DistilBERT

### [Vector]Model

This class type does not need a document set to be instantiated (with the exception of TFIDFModel). To instantiated the model simply use `[Vector]Model` (for example, `Word2VecModel()`). You can then use the model itself as a function call to generate a vector for a tokenized sentence.

```python
model = Word2VecModel()
model('i like to eat cheese')
>>> [-0.15497589  0.11981201 -0.02108765 ...  0.0141983  -0.12994385 0.18857574]
```

The TFIDFModel class, however, requires a document set in order to instantiate the model.

```python
docset, indices = docset_loader('D1001A-A', 'data/devtest.json')
model = TFIDFModel(docset)
model('i like to eat cheese')
>>> [0., 0., 0., ..., 0., 0., 0.] # the vector is sparse since it is the length of the vocabulary
```

### DocumentTo[Model]

This class type needs documents to be instantiated. It takes the documents and return a class that can be used to reference word vectors within a document or create a similarity matrix.

Note that the `__getitem__` method depends on whether the class is treating sentences or articles as documents. If articles are documents: the _key argument should be {docset_id}.{doc_id} (e.g., D1001A-A.APW19990421.0284). If sentences are documents: the _key argument should_key should be {docset_id}.{sentence_index} (e.g., D1001A-A.0). You can also always reference the data using an integer index.

You can load DocumentTo[Model] classes directly from the data with `from_data`.

```python
docset_vectors = DocumentToDistilBert.from_data('D1001A-A', 'data/devtest.json')
```

```python
eval_docs_sentences, _ = docset_loader(
    'D1001A-A', 'data/devtest.json', sentences_are_documents=True
)
docset_vectors = DocumentToTFIDF.from_data(
    'D1001A-A', 'data/devtest.json', sentences_are_documents=True, eval_documents=eval_docs_s
)
```